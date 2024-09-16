import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil

def print_system_info():
    print("\n--- Información del Sistema ---")
    print(f"Núcleos CPU: {multiprocessing.cpu_count()}")
    print(f"Memoria Total: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"Memoria Disponible: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
    
    if torch.cuda.is_available():
        print("\n--- Información de GPU ---")
        print(f"GPU Disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU Total: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"Memoria GPU Disponible: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("\nNo se detectó GPU compatible con CUDA.")
class PixelArtDataset(Dataset):
    def __init__(self, csv_file, transform=None, max_frames=60):
        self.data = pd.read_csv(csv_file, encoding='utf-8')
        print("Primeras filas del CSV:")
        print(self.data.head())
        print("\nColumnas del CSV:")
        print(self.data.columns)
        if 'file_path' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("El CSV debe contener las columnas 'file_path' y 'label'")
        self.transform = transform
        self.max_frames = max_frames
        self.animations = self.data.groupby('label')
    
    def __len__(self):
        return len(self.animations)
    
    def __getitem__(self, idx):
        label = list(self.animations.groups.keys())[idx]
        animation_frames = self.animations.get_group(label)
        
        frames = []
        for _, row in animation_frames.iterrows():
            img_path = row['file_path']
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            
            if len(frames) >= self.max_frames:
                break
        
        if len(frames) < self.max_frames:
            frames.extend([torch.zeros_like(frames[0]) for _ in range(self.max_frames - len(frames))])
        elif len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        
        return torch.stack(frames), label

class TextToAnimationCNN(nn.Module):
    def __init__(self, text_embed_size, hidden_size, max_frames, frame_size):
        super(TextToAnimationCNN, self).__init__()
        self.text_embedding = nn.Embedding(1000, text_embed_size)
        self.lstm = nn.LSTM(text_embed_size, hidden_size, batch_first=True)
        
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.frame_generator = nn.Linear(hidden_size, max_frames * hidden_size)
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.hidden_size = hidden_size
    
    def forward(self, text):
        embedded = self.text_embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        
        frame_features = self.frame_generator(hidden.squeeze(0))
        frame_features = frame_features.view(-1, self.max_frames, self.hidden_size)
        
        frames = []
        for i in range(self.max_frames):
            frame = self.conv_transpose(frame_features[:, i, :].unsqueeze(-1).unsqueeze(-1))
            frames.append(frame)
        
        return torch.stack(frames, dim=1)

def process_batch(model, batch_frames, batch_labels, criterion, device):
    batch_frames = batch_frames.to(device)
    text_indices = torch.tensor([hash(label) % 1000 for label in batch_labels], device=device)
    
    outputs = model(text_indices)
    loss = criterion(outputs, batch_frames)
    
    return loss

def train_model(model, train_loader, num_epochs, device, model_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = len(train_loader)
        print(f"\nÉpoca [{epoch+1}/{num_epochs}]")
        
        for i, (batch_frames, batch_labels) in enumerate(train_loader):
            loss = process_batch(model, batch_frames, batch_labels, criterion, device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            progress = (i + 1) / batch_count
            print(f"\rProgreso: [{'=' * int(50 * progress):{50}}] {progress:.1%}", end='')
        
        avg_loss = total_loss / batch_count
        elapsed_time = time.time() - start_time
        print(f"\nPérdida promedio: {avg_loss:.4f}, Tiempo transcurrido: {elapsed_time:.2f} segundos")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{model_path}_epoch_{epoch+1}.pth")
            print(f"Modelo guardado en {model_path}_epoch_{epoch+1}.pth")
        
        # Imprimir información de uso de recursos cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print("\n--- Uso de Recursos ---")
            print(f"CPU: {psutil.cpu_percent()}%")
            print(f"Memoria: {psutil.virtual_memory().percent}%")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB / {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    
    torch.save(model.state_dict(), f"{model_path}_final.pth")
    print(f"Modelo final guardado en {model_path}_final.pth")
    print(f"Tiempo total de entrenamiento: {(time.time() - start_time) / 60:.2f} minutos")

def generate_animation(model, text, device, max_frames):
    model.eval()
    with torch.no_grad():
        text_index = torch.tensor([hash(text) % 1000]).to(device)
        generated_frames = model(text_index)
    return generated_frames.squeeze(0).cpu().numpy()

def save_animation(frames, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame = ((frame * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(frame)
        img.save(output_path / f"frame_{i:04d}.png")
    print(f"Animación guardada en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo o genera una animación de pixel art basada en texto.")
    parser.add_argument("--train", action="store_true", help="Entrenar el modelo")
    parser.add_argument("--generate", action="store_true", help="Generar una animación")
    parser.add_argument("--text", type=str, help="Texto para generar la animación")
    parser.add_argument("--output", type=str, default="output_animation", help="Carpeta de salida para la animación")
    parser.add_argument("--model_path", type=str, default="model", help="Ruta base para guardar/cargar el modelo")
    parser.add_argument("--csv_file", type=str, default="dataset.csv", help="Archivo CSV con los datos de entrenamiento")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas para entrenar")
    args = parser.parse_args()

    print("Iniciando el script...")
    print(f"Modo: {'Entrenamiento' if args.train else 'Generación'}")

    print_system_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}")
    
    max_frames = 60
    frame_size = 64
    
    print("Inicializando el modelo...")
    model = TextToAnimationCNN(text_embed_size=100, hidden_size=256, max_frames=max_frames, frame_size=frame_size).to(device)
    print("Modelo inicializado.")

    if args.train:
        print(f"Cargando datos desde {args.csv_file}...")
        transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            dataset = PixelArtDataset(csv_file=args.csv_file, 
                                      transform=transform, 
                                      max_frames=max_frames)
            print(f"Dataset cargado. Tamaño del dataset: {len(dataset)}")
            
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=multiprocessing.cpu_count())
            print("DataLoader creado. Iniciando entrenamiento...")
            
            train_model(model, dataloader, num_epochs=args.epochs, device=device, model_path=args.model_path)
        except Exception as e:
            print(f"Error al cargar o procesar el dataset: {e}")
            return
    elif args.generate:
        if not os.path.exists(f"{args.model_path}_final.pth"):
            print(f"Error: No se encontró el modelo en {args.model_path}_final.pth")
            print("Por favor, entrena el modelo primero usando el argumento --train")
            return
        
        model.load_state_dict(torch.load(f"{args.model_path}_final.pth"))
        print(f"Modelo cargado desde {args.model_path}_final.pth")

        if args.text:
            text = args.text
        else:
            text = input("Ingrese el texto para generar la animación: ")
        
        new_animation = generate_animation(model, text, device, max_frames)
        save_animation(new_animation, args.output)
    else:
        print("Por favor, especifica --train para entrenar el modelo o --generate para generar una animación.")

    print("Script finalizado.")

if __name__ == "__main__":
    main()