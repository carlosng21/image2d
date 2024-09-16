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
import json
import matplotlib.pyplot as plt

class PixelArtDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=60):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_frames = max_frames
        self.animations = self._load_animations()
    
    def _load_animations(self):
        return [d for d in self.root_dir.iterdir() if d.is_dir()]
    
    def __len__(self):
        return len(self.animations)
    
    def __getitem__(self, idx):
        anim_dir = self.animations[idx]
        
        with open(anim_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        attributes = metadata["attributes"]
        description = metadata["description"]
        
        frames = []
        for i in range(self.max_frames):
            img_path = anim_dir / f"frame_{i:04d}.png"
            if img_path.exists():
                image = Image.open(img_path).convert('RGBA')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            else:
                break
        
        if len(frames) < self.max_frames:
            frames.extend([torch.zeros_like(frames[0]) for _ in range(self.max_frames - len(frames))])
        elif len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        
        return torch.stack(frames), attributes, description

def visualize_batch(batch_frames, batch_num):
    batch_frames = batch_frames.cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        if i < batch_frames.shape[0]:
            frame = (batch_frames[i] * 255).astype(np.uint8).transpose(1, 2, 0)
            ax.imshow(frame)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'batch_visualization_{batch_num}.png')
    plt.close()

def print_dataset_info(dataset):
    print(f"Total number of animations: {len(dataset)}")
    print("First 5 animation folders:")
    for anim_dir in dataset.animations[:5]:
        print(f"  - {anim_dir}")
        print(f"    Number of frames: {len(list(anim_dir.glob('frame_*.png')))}")
        with open(anim_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"    Attributes: {metadata['attributes']}")
        print(f"    Description: {metadata['description']}")
    print("\n")

class TextToAnimationCNN(nn.Module):
    def __init__(self, text_embed_size, hidden_size, max_frames, frame_size):
        super(TextToAnimationCNN, self).__init__()
        self.text_embedding = nn.Embedding(1000, text_embed_size)
        self.lstm = nn.LSTM(text_embed_size, hidden_size, batch_first=True)
        
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
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
        
        base_frame = self.conv_transpose(frame_features[:, 0, :].unsqueeze(-1).unsqueeze(-1))
        
        frames = [base_frame]
        for i in range(1, self.max_frames):
            delta = self.conv_transpose(frame_features[:, i, :].unsqueeze(-1).unsqueeze(-1))
            new_frame = torch.clamp(frames[-1] + delta, 0, 1)
            frames.append(new_frame)
        
        return torch.stack(frames, dim=1)

def train_model(model, train_loader, num_epochs, device, model_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = len(train_loader)
        print(f"\nÉpoca [{epoch+1}/{num_epochs}]")
        for i, (batch_frames, batch_attributes, batch_descriptions) in enumerate(train_loader):
            batch_frames = batch_frames.to(device)
            text_indices = torch.tensor([hash(" ".join(attr) + desc) % 1000 for attr, desc in zip(batch_attributes, batch_descriptions)]).to(device)
            
            optimizer.zero_grad()
            outputs = model(text_indices)
            loss = criterion(outputs, batch_frames)
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
        frame = (frame * 255).astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(frame, 'RGBA')
        img.save(output_path / f"frame_{i:04d}.png")
    print(f"Animación guardada en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo o genera una animación de pixel art basada en texto.")
    parser.add_argument("--train", action="store_true", help="Entrenar el modelo")
    parser.add_argument("--generate", action="store_true", help="Generar una animación")
    parser.add_argument("--text", type=str, help="Texto para generar la animación")
    parser.add_argument("--output", type=str, default=r"C:\\Users\\camlo\\Desktop\\intento\\animacion", help="Directorio de salida para la animación")
    parser.add_argument("--model_path", type=str, default=r"C:\\Users\\camlo\\Desktop\\intento", help="Ruta base para guardar/cargar el modelo")
    parser.add_argument("--data_dir", type=str, default=r"C:\\Users\\camlo\\Desktop\\intento\\animacion", help="Directorio con los frames procesados")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas para entrenar")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_frames = 60
    frame_size = 64
    
    model = TextToAnimationCNN(text_embed_size=100, hidden_size=256, max_frames=max_frames, frame_size=frame_size).to(device)

    if args.train:
        transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
        ])
        
        dataset = PixelArtDataset(root_dir=args.data_dir, 
                                  transform=transform, 
                                  max_frames=max_frames)
        
        print_dataset_info(dataset)
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        train_model(model, dataloader, num_epochs=args.epochs, device=device, model_path=args.model_path)
    elif args.generate:
        if not os.path.exists(f"{args.model_path}_final.pth"):
            print(f"Error: No se encontró el modelo en {args.model_path}_final.pth")
            return
        
        model.load_state_dict(torch.load(f"{args.model_path}_final.pth"))

        if args.text:
            text = args.text
        else:
            text = input("Ingrese el texto para generar la animación: ")
        
        new_animation = generate_animation(model, text, device, max_frames)
        save_animation(new_animation, args.output)
    else:
        print("Por favor, especifica --train para entrenar el modelo o --generate para generar una animación.")

if __name__ == "__main__":
    main()