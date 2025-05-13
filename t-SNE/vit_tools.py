import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


def init_vit_model(device):
    weights = MViT_V2_S_Weights.DEFAULT
    model = mvit_v2_s(weights=weights)
    model.head = torch.nn.Identity()
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


def get_video_feats(video_path, model, preprocess, device):
    frames = []
    pic_names = os.listdir(video_path)
    for pic_name in pic_names:
        pic_path = os.path.join(video_path, pic_name)
        frame = cv2.imread(pic_path)
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        frames.append(frame)
        frames.append(frame)
    if len(frames) >= 16:
        frames = frames[:16]
    vid_tube = torch.stack(frames, dim=0)
    batch = preprocess(vid_tube).unsqueeze(0)
    batch = batch.to(device).float()
    with torch.no_grad():
        # print(batch.size())
        features = model(batch)
        # print(features.size())
        features = torch.squeeze(features).cpu().numpy()
    return features


def get_videos_vit_feats(video_paths, model, preprocess, device):
    video_feats = []
    for index in tqdm(range(len(video_paths))):
        video_feats.append(get_video_feats(video_paths[index], model, preprocess, device))
    video_feats = np.vstack(video_feats)
    return video_feats
