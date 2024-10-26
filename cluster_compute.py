import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from sklearn.cluster import KMeans

def load_images(folder_path):
    # 按照文件名中的数字部分对文件进行排序
    image_files = os.listdir(folder_path)
    image_files.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))

    images = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img, img_path))
    return images

def extract_features(frames):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)  # 使用 GPU 运行模型
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for frame, _ in frames:
            input_tensor = preprocess(frame).unsqueeze(0).to(device)
            feature = model(input_tensor).squeeze().cpu().numpy()
            features.append(feature)
    return np.array(features)

def cluster_key_frames(features, required_key_frames):
    kmeans = KMeans(n_clusters=required_key_frames)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_

    key_frame_indices = []
    for center in cluster_centers:
        distances = np.linalg.norm(features - center, axis=1)
        closest_index = np.argmin(distances)
        key_frame_indices.append(closest_index)

    return sorted(set(key_frame_indices))

def save_key_frames(images, key_frame_indices, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for index in key_frame_indices:
        img, img_path = images[index]
        save_path = os.path.join(save_folder, os.path.basename(img_path))
        cv2.imwrite(save_path, img)

def extract_key_frames(folder_path, required_key_frames=8, save_folder=None):
    images = load_images(folder_path)
    if len(images) < required_key_frames:
        print("Warning: Number of frames is less than required key frames.")
        return []

    features = extract_features(images)
    key_frame_indices = cluster_key_frames(features, required_key_frames)

    if save_folder:
        save_key_frames(images, key_frame_indices, save_folder)

    return key_frame_indices

# 指定根文件夹路径
root_folder = 'activity_256x256q5'
required_key_frames = 4

# 遍历文件夹
for subdir, dirs, files in os.walk(root_folder):
    if any(file.endswith(('.jpg', '.png', '.jpeg')) for file in files):
        new_subdir = os.path.join('clustered_keyframes', os.path.relpath(subdir, root_folder))
        key_frame_indices = extract_key_frames(subdir, required_key_frames, new_subdir)
        if key_frame_indices:
            print(f"Selected keyframe indices in {subdir}: {key_frame_indices}")
        else:
            print(f"No keyframes extracted in {subdir}.")

