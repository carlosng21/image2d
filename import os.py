

import os
import pandas as pd

# Ruta a la carpeta que contiene las subcarpetas con imágenes
base_dir = r'C:\Users\camlo\Desktop\intento\processed_frames'

# Lista para almacenar la información del dataset
data = []

# Recorre todas las carpetas en la ruta base
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # Recorre todos los archivos en la subcarpeta
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Añade la ruta del archivo y la etiqueta (nombre de la carpeta) al dataset
                data.append({'file_path': file_path, 'label': folder})

# Crea un DataFrame de pandas
df = pd.DataFrame(data)

# Guarda el DataFrame en un archivo CSV
csv_path = r'C:\Users\camlo\Desktop\intento\dataset.csv'
df.to_csv(csv_path, index=False)

print(f'Dataset guardado en {csv_path}')