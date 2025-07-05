import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict

# === Config ===
metadata_path = "HAM10000_metadata.csv"
images_path = "images"
output_root = "HAM10000"
selected_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
samples_per_class = 100  # total subset size ~700 images
num_clients = 6
test_split = 0.15
random_seed = 42

# === Load and filter metadata ===
df = pd.read_csv(metadata_path)
df = df[df['dx'].isin(selected_classes)]

# Subset: balanced small sample per class
df_subset = df.groupby('dx').apply(lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=random_seed))
df_subset.reset_index(drop=True, inplace=True)

# === Split global test set ===
df_trainval, df_test = train_test_split(df_subset, test_size=test_split, stratify=df_subset['dx'], random_state=random_seed)

# === Split training set across clients ===
client_splits = [[] for _ in range(num_clients)]

for _, row in df_trainval.iterrows():
    # Round-robin by class to each client
    dx = row['dx']
    assigned = hash(row['image_id']) % num_clients
    client_splits[assigned].append(row)

# === Save data ===
def save_client_data(client_id, rows):
    client_dir = os.path.join(output_root, f"client_{client_id+1}")
    img_dir = os.path.join(client_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    label_rows = []
    for row in rows:
        img_id = row['image_id'] + ".jpg"
        src_path = os.path.join(images_path, img_id)
        dst_path = os.path.join(img_dir, img_id)
        shutil.copyfile(src_path, dst_path)
        label_rows.append({'image': img_id, 'label': row['dx']})
    
    # Save CSV
    pd.DataFrame(label_rows).to_csv(os.path.join(client_dir, "labels.csv"), index=False)

# Save for each client
for i in range(num_clients):
    save_client_data(i, client_splits[i])

# === Save global test set ===
test_dir = os.path.join(output_root, "test")
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)

test_labels = []
for _, row in df_test.iterrows():
    img_id = row['image_id'] + ".jpg"
    src_path = os.path.join(images_path, img_id)
    dst_path = os.path.join(test_dir, "images", img_id)
    shutil.copyfile(src_path, dst_path)
    test_labels.append({'image': img_id, 'label': row['dx']})

pd.DataFrame(test_labels).to_csv(os.path.join(test_dir, "labels.csv"), index=False)

print(f"✅ Distributed HAM10000 into {num_clients} clients and 1 test set.")
