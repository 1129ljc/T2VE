import os
import cv2
import random
import torch
import numpy as np
import albumentations
import torch.nn as nn
from tqdm import tqdm
from transformers import XCLIPVisionModel


class XCLIP_DeMamba(nn.Module):
    def __init__(self):
        super(XCLIP_DeMamba, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained("/data2/ljc/code/xclip-base-patch32/")

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        # last_hidden_state = outputs['last_hidden_state']
        pooler_output = outputs['pooler_output']
        # hidden_states = outputs['hidden_states']
        global_feat = pooler_output.reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        return global_feat


def init_xclip_model(device):
    video_encoder = XCLIP_DeMamba().to(device)
    aug_list = [
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ]
    trans = albumentations.Compose(aug_list)
    return video_encoder, trans


def get_video_feats(video_path, model, trans, device):
    pic_names = sorted(os.listdir(video_path))
    select_frame_nums = 8
    if len(pic_names) >= select_frame_nums:
        start_frame = random.randint(0, len(pic_names) - select_frame_nums)
        select_frames = pic_names[start_frame:start_frame + select_frame_nums]
        frames = []
        for frame_index in range(len(select_frames)):
            pic_name = select_frames[frame_index]
            frame_path = os.path.join(video_path, pic_name)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = trans(image=frame)["image"]
            frames.append(frame.transpose(2, 0, 1)[np.newaxis, :])
    else:
        pad_num = select_frame_nums - len(pic_names)
        frames = []
        for frame_index in range(len(pic_names)):
            pic_name = pic_names[frame_index]
            frame_path = os.path.join(video_path, pic_name)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = trans(image=frame)["image"]
            frames.append(frame.transpose(2, 0, 1)[np.newaxis, :])
        for i in range(pad_num):
            frames.append(np.zeros(shape=(1, 3, 224, 224)))
    frames = np.concatenate(frames, 0)
    frames = torch.tensor(frames)
    frames = torch.unsqueeze(frames, dim=0).to(device)
    with torch.no_grad():
        output_t = model(frames)
        output_t = torch.squeeze(output_t).cpu().numpy()
    return output_t


def get_videos_xclip_feats(video_paths, model, trans, device):
    video_feats = []
    for index in tqdm(range(len(video_paths))):
        video_feats.append(get_video_feats(video_paths[index], model, trans, device))
    video_feats = np.vstack(video_feats)
    return video_feats
