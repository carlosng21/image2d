import os
import pandas as pd

base_dir = r'C:\Users\camlo\Desktop\intento\processed_frames'
csv_labels_path = r'C:\Users\camlo\Desktop\intento\labels.csv'

# Leer el archivo CSV con las etiquetas
if os.path.isfile(csv_labels_path):
    labels_df = pd.read_csv(csv_labels_path)
    labels_dict = dict(zip(labels_df['animation_name'], labels_df['label']))
else:
    labels_dict = {}  # Si el archivo CSV no existe, usar un diccionario vacío

data = []

try:
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Usar el nombre de la carpeta como nombre de animación para obtener la etiqueta
                    animation_name = folder
                    label = labels_dict.get(animation_name, 'Unknown')  # Usa 'Unknown' si no se encuentra la etiqueta
                    data.append({'animation_name': animation_name, 'label': label})
except Exception as e:
    print(f'Ocurrió un error: {e}')

df = pd.DataFrame(data)
csv_path = r'C:\Users\camlo\Desktop\intento\dataset.csv'
df.to_csv(csv_path, index=False)

print(f'Dataset guardado en {csv_path}')