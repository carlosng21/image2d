#intento.py

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import imageio
from PIL import Image
import os
import json
import re

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

def analyze_examples(examples_dir):
    examples_dir = Path(examples_dir)
    all_attributes = set()
    categories = set()
    animation_types = set()
    
    for file in examples_dir.glob('*'):
        if file.is_file():
            with open(file, 'r') as f:
                content = f.read()
            
            # Extract attributes and descriptions from JSON-like strings
            json_matches = re.findall(r'\{[^}]+\}', content)
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if 'attributes' in data:
                        all_attributes.update(data['attributes'])
                except json.JSONDecodeError:
                    pass
            
            # Extract attributes from list-like strings
            list_matches = re.findall(r'\[[^\]]+\]', content)
            for match in list_matches:
                try:
                    data = eval(match)
                    if isinstance(data, list):
                        all_attributes.update(data)
                except:
                    pass
    
    # Categorize attributes
    for attr in all_attributes:
        if attr in ['personaje', 'objeto', 'enemigo', 'animal']:
            categories.add(attr)
        elif attr in ['ataque', 'idle', 'caminar', 'correr', 'saltar', 'morir']:
            animation_types.add(attr)
    
    return list(all_attributes), list(categories), list(animation_types)

def get_user_defined_categories(initial_categories):
    print("Categorías actuales:", ", ".join(initial_categories))
    while True:
        action = input("¿Desea añadir (A), eliminar (E) o finalizar (F)? ").strip().upper()
        if action == 'A':
            category = input("Ingrese una nueva categoría: ").strip()
            if category and category not in initial_categories:
                initial_categories.append(category)
        elif action == 'E':
            category = input("Ingrese la categoría a eliminar: ").strip()
            if category in initial_categories:
                initial_categories.remove(category)
        elif action == 'F':
            break
    return initial_categories

def get_user_defined_animation_types(initial_types):
    print("Tipos de animación actuales:", ", ".join(initial_types))
    while True:
        action = input("¿Desea añadir (A), eliminar (E) o finalizar (F)? ").strip().upper()
        if action == 'A':
            anim_type = input("Ingrese un nuevo tipo de animación: ").strip()
            if anim_type and anim_type not in initial_types:
                initial_types.append(anim_type)
        elif action == 'E':
            anim_type = input("Ingrese el tipo de animación a eliminar: ").strip()
            if anim_type in initial_types:
                initial_types.remove(anim_type)
        elif action == 'F':
            break
    return initial_types

def get_attributes_from_user(file_name, categories, animation_types, all_attributes):
    print(f"\nProcesando: {file_name}")
    print("Ingrese los atributos para este asset. Puede seleccionar múltiples opciones separadas por comas.")
    
    attributes = []
    
    print("Categorías principales (separadas por comas):", ", ".join(categories))
    main_selection = input("Seleccione las categorías principales: ").lower().split(',')
    attributes.extend([cat.strip() for cat in main_selection if cat.strip() in categories])
    
    print("\nTipos de animación (separados por comas):", ", ".join(animation_types))
    animation_selection = input("Seleccione los tipos de animación: ").lower().split(',')
    attributes.extend([anim.strip() for anim in animation_selection if anim.strip() in animation_types])
    
    print("\nAtributos adicionales disponibles:", ", ".join(all_attributes))
    print("Ingrese atributos adicionales separados por comas (o ingrese nuevos atributos):")
    additional_attributes = input("Atributos adicionales: ").lower().split(',')
    attributes.extend([attr.strip() for attr in additional_attributes if attr.strip()])
    
    return list(set(attributes))

def get_description_from_user(file_name):
    print(f"\nProcesando: {file_name}")
    description = input("Ingrese una breve descripción para este asset: ").strip()
    return description

def process_file(file_path, output_dir, categories, animation_types, all_attributes):
    frames = extract_frames(file_path)
    processed_frames = [remove_background_improved(frame) for frame in frames]
    
    file_name = file_path.stem
    attributes = get_attributes_from_user(file_name, categories, animation_types, all_attributes)
    description = get_description_from_user(file_name)
    
    file_output_dir = output_dir / file_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(processed_frames):
        imageio.imwrite(str(file_output_dir / f"frame_{i:04d}.png"), frame)
    
    metadata = {
        "attributes": attributes,
        "description": description
    }
    
    with open(file_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return attributes, description

def create_tf_dataset(data_dir):
    data_dir = Path(data_dir)
    asset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    def load_asset(asset_dir):
        with open(asset_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        attributes = metadata["attributes"]
        description = metadata["description"]
        
        frames = list((asset_dir).glob('frame_*.png'))
        frames.sort()
        
        def load_image(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=4)
            img = tf.image.resize(img, [64, 64])
            return img
        
        images = tf.stack([load_image(str(frame)) for frame in frames])
        return images, attributes, description
    
    dataset = tf.data.Dataset.from_tensor_slices([str(d) for d in asset_dirs])
    dataset = dataset.map(lambda x: tf.py_function(load_asset, [x], [tf.float32, tf.string, tf.string]), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def process_all_files(assets_dir, output_dir, examples_dir):
    assets_dir = Path(assets_dir)
    output_dir = Path(output_dir)
    
    supported_formats = ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.gif', '.webp', '.png', '.jpg', '.jpeg']
    
    print("Analizando ejemplos y creando categorías iniciales...")
    all_attributes, categories, animation_types = analyze_examples(examples_dir)
    
    print("\nDefinición de categorías y tipos de animación:")
    categories = get_user_defined_categories(categories)
    animation_types = get_user_defined_animation_types(animation_types)
    
    all_metadata = {}
    for file_path in assets_dir.iterdir():
        if file_path.suffix.lower() in supported_formats:
            attributes, description = process_file(file_path, output_dir, categories, animation_types, all_attributes)
            all_metadata[file_path.name] = {"attributes": attributes, "description": description}
    
    print("\nResumen de assets, atributos y descripciones:")
    for file_name, metadata in all_metadata.items():
        print(f"\n{file_name}:")
        print(f"  Atributos: {', '.join(metadata['attributes'])}")
        print(f"  Descripción: {metadata['description']}")

def find_assets_by_search(dataset, search_term):
    def match_search(attributes, description):
        attributes_str = " ".join(attributes.numpy().decode('utf-8').split(','))
        description_str = description.numpy().decode('utf-8')
        return search_term.lower() in attributes_str.lower() or search_term.lower() in description_str.lower()
    
    return dataset.filter(lambda images, attributes, description: tf.py_function(match_search, [attributes, description], tf.bool))

assets_dir = Path("assets")
output_dir = Path("processed_frames")
examples_dir = Path("ejemplos")

process_all_files(assets_dir, output_dir, examples_dir)

tf_dataset = create_tf_dataset(output_dir)

print(f"\nDataset creado con {tf_dataset.cardinality().numpy()} assets.")

for images, attributes, description in tf_dataset.take(1):
    print("Forma de las imágenes:", images.shape)
    print("Tipo de datos de las imágenes:", images.dtype)
    print("Atributos:", attributes.numpy())
    print("Descripción:", description.numpy())

search_term = input("Ingrese un término de búsqueda: ")
filtered_dataset = find_assets_by_search(tf_dataset, search_term)
print(f"\nAssets encontrados con el término de búsqueda '{search_term}': {filtered_dataset.cardinality().numpy()}")

for images, attributes, description in filtered_dataset.take(5):
    print(f"Atributos: {attributes.numpy().decode('utf-8')}")
    print(f"Descripción: {description.numpy().decode('utf-8')}")
    print("---")