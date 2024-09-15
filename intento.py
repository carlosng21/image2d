import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import imageio
from PIL import Image
import os
import json

def extract_frames(file_path):
    file_path = Path(file_path)
    frames = []
    
    if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.flv', '.wmv']:
        video = cv2.VideoCapture(str(file_path))
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()
    elif file_path.suffix.lower() in ['.gif', '.webp']:
        gif = imageio.get_reader(file_path)
        for frame in gif:
            frames.append(frame)
    else:
        img = Image.open(file_path)
        frames.append(np.array(img))
    
    return frames

def remove_background_improved(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    background_color = np.unravel_index(hist.argmax(), hist.shape)
    
    lower_bound = np.array([max(0, background_color[0] - 10), max(0, background_color[1] - 40), 0])
    upper_bound = np.array([min(180, background_color[0] + 10), min(255, background_color[1] + 40), 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    mask = cv2.bitwise_not(mask)
    
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    
    return rgba

def get_category_from_user(file_name):
    print(f"\nProcesando: {file_name}")
    category = input("Ingrese una categoría para este asset (ej. personaje, objeto, enemigo): ").strip()
    return category

def get_attributes_from_user(file_name):
    print(f"\nProcesando: {file_name}")
    print("Ingrese los atributos para este asset. Puede seleccionar múltiples opciones separadas por comas.")
    
    attributes = []
    
    main_categories = ['personaje', 'objeto', 'enemigo', 'escenario']
    print("Categorías principales (separadas por comas):", ", ".join(main_categories))
    main_selection = input("Seleccione las categorías principales: ").lower().split(',')
    attributes.extend([cat.strip() for cat in main_selection if cat.strip() in main_categories])
    
    animation_types = ['idle', 'caminar', 'correr', 'saltar', 'atacar', 'morir']
    print("\nTipos de animación (separados por comas):", ", ".join(animation_types))
    animation_selection = input("Seleccione los tipos de animación: ").lower().split(',')
    attributes.extend([anim.strip() for anim in animation_selection if anim.strip() in animation_types])
    
    print("\nIngrese atributos adicionales separados por comas (ej. color, tamaño, estilo):")
    additional_attributes = input("Atributos adicionales: ").lower().split(',')
    attributes.extend([attr.strip() for attr in additional_attributes if attr.strip()])
    
    return list(set(attributes))

def process_file(file_path, output_dir):
    frames = extract_frames(file_path)
    processed_frames = [remove_background_improved(frame) for frame in frames]
    
    file_name = file_path.stem
    attributes = get_attributes_from_user(file_name)
    
    file_output_dir = output_dir / file_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(processed_frames):
        imageio.imwrite(str(file_output_dir / f"frame_{i:04d}.png"), frame)
    
    with open(file_output_dir / "attributes.json", "w") as f:
        json.dump(attributes, f)
    
    return attributes

def create_tf_dataset(data_dir):
    data_dir = Path(data_dir)
    asset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    def load_asset(asset_dir):
        with open(asset_dir / "attributes.json", "r") as f:
            attributes = json.load(f)
        
        frames = list((asset_dir).glob('frame_*.png'))
        frames.sort()
        
        def load_image(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=4)
            img = tf.image.resize(img, [64, 64])
            return img
        
        images = tf.stack([load_image(str(frame)) for frame in frames])
        return images, attributes
    
    dataset = tf.data.Dataset.from_tensor_slices([str(d) for d in asset_dirs])
    dataset = dataset.map(lambda x: tf.py_function(load_asset, [x], [tf.float32, tf.string]), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def process_all_files(assets_dir, output_dir):
    assets_dir = Path(assets_dir)
    output_dir = Path(output_dir)
    
    supported_formats = ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.gif', '.webp', '.png', '.jpg', '.jpeg']
    
    all_attributes = {}
    for file_path in assets_dir.iterdir():
        if file_path.suffix.lower() in supported_formats:
            attributes = process_file(file_path, output_dir)
            all_attributes[file_path.name] = attributes
    
    print("\nResumen de assets y atributos:")
    for file_name, attributes in all_attributes.items():
        print(f"\n{file_name}:")
        print(f"  Atributos: {', '.join(attributes)}")

assets_dir = Path("assets")
output_dir = Path("processed_frames")

process_all_files(assets_dir, output_dir)

tf_dataset = create_tf_dataset(output_dir)

print(f"\nDataset creado con {tf_dataset.cardinality().numpy()} assets.")

for images, attributes in tf_dataset.take(1):
    print("Forma de las imágenes:", images.shape)
    print("Tipo de datos de las imágenes:", images.dtype)
    print("Atributos:", attributes.numpy())

def find_assets_by_attributes(dataset, required_attributes):
    def match_attributes(attributes):
        return all(attr in attributes for attr in required_attributes)
    
    return dataset.filter(lambda images, attributes: tf.py_function(match_attributes, [attributes], tf.bool))

required_attributes = ['personaje', 'atacar']
filtered_dataset = find_assets_by_attributes(tf_dataset, required_attributes)
print(f"\nAssets encontrados con los atributos {required_attributes}: {filtered_dataset.cardinality().numpy()}")