import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEModel


def init_videomae_model(device):
    model = VideoMAEModel.from_pretrained("/data2/ljc/code/VideoMAE/videomae-base/")
    model = model.to(device)
    model.eval()
    return model


def get_video_feats(video_path, model, device):
    frames = []
    pic_names = os.listdir(video_path)
    for pic_name in pic_names:
        pic_path = os.path.join(video_path, pic_name)
        frame = cv2.imread(pic_path)
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.from_numpy(frame)
        frames.append(frame)
        frames.append(frame)
    if len(frames) >= 16:
        frames = frames[:16]
    batch = torch.stack(frames, dim=0).unsqueeze(0)
    batch = batch.to(device).float()
    with torch.no_grad():
        features = model(batch)
        features = torch.mean(torch.squeeze(features['last_hidden_state']), dim=0)
        features = features.cpu().numpy()
    return features


def get_videos_videomae_feats(video_paths, model, device):
    video_feats = []
    for index in tqdm(range(len(video_paths))):
        video_feats.append(get_video_feats(video_paths[index], model, device))
    video_feats = np.vstack(video_feats)
    return video_feats