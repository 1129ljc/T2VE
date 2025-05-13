import os
import random
import torch
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_video_feats(video_path, model, preprocess, device):
    pic_names = os.listdir(video_path)
    selected_file = random.choice(pic_names)
    pic_path = os.path.join(video_path, selected_file)
    image = preprocess(Image.open(pic_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = torch.squeeze(image_features).cpu().numpy()
    return image_features


def get_videos_clip_feats(video_paths, model, preprocess, device):
    video_feats = []
    for index in tqdm(range(len(video_paths))):
        video_feats.append(get_video_feats(video_paths[index], model, preprocess, device))
    video_feats = np.vstack(video_feats)
    return video_feats


def init_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess
