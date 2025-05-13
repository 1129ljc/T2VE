import os
import cv2
import random
import albumentations
import numpy as np
import torch
from torch.utils.data import Dataset

train_index_file_dir = '/data2/ljc/dataset/GenVideo_frame/split/train/'
val_index_file_dir = '/data2/ljc/dataset/GenVideo_frame/split/val/'
test_index_file_dir = '/data2/ljc/dataset/GenVideo_frame/GenVideo-Test/'


class GenVideoTrain(Dataset):
    def __init__(self):
        self.select_frame_nums = 8
        self.select_video_nums = 512000

        # get video txt index file path
        K400_txt_file_path = os.path.join(train_index_file_dir, 'k400.txt')
        Youku_1M_10s_txt_file_path = os.path.join(train_index_file_dir, 'Youku_1M_10s.txt')
        ZeroScope_txt_file_path = os.path.join(train_index_file_dir, 'ZeroScope.txt')
        I2VGEN_XL_txt_file_path = os.path.join(train_index_file_dir, 'I2VGEN_XL.txt')
        SVD_txt_file_path = os.path.join(train_index_file_dir, 'SVD.txt')
        Latte_txt_file_path = os.path.join(train_index_file_dir, 'Latte.txt')
        OpenSora_txt_file_path = os.path.join(train_index_file_dir, 'OpenSora.txt')
        pika_txt_file_path = os.path.join(train_index_file_dir, 'pika.txt')
        SD_txt_file_path = os.path.join(train_index_file_dir, 'SD.txt')
        SEINE_txt_file_path = os.path.join(train_index_file_dir, 'SEINE.txt')
        DynamicCrafter_txt_file_path = os.path.join(train_index_file_dir, 'DynamicCrafter.txt')
        VideoCrafter_txt_file_path = os.path.join(train_index_file_dir, 'VideoCrafter.txt')

        # read video txt index file
        K400_video_names = self.read_txt(K400_txt_file_path, label=0)
        Youku_1M_10s_video_names = self.read_txt(Youku_1M_10s_txt_file_path, label=0)
        ZeroScope_video_names = self.read_txt(ZeroScope_txt_file_path, label=1)
        I2VGEN_XL_video_names = self.read_txt(I2VGEN_XL_txt_file_path, label=1)
        SVD_video_names = self.read_txt(SVD_txt_file_path, label=1)
        Latte_video_names = self.read_txt(Latte_txt_file_path, label=1)
        OpenSora_video_names = self.read_txt(OpenSora_txt_file_path, label=1)
        pika_video_names = self.read_txt(pika_txt_file_path, label=1)
        SD_video_names = self.read_txt(SD_txt_file_path, label=1)
        SEINE_video_names = self.read_txt(SEINE_txt_file_path, label=1)
        DynamicCrafter_video_names = self.read_txt(DynamicCrafter_txt_file_path, label=1)
        VideoCrafter_video_names = self.read_txt(VideoCrafter_txt_file_path, label=1)

        # group real and fake index set
        real_video_names = K400_video_names + Youku_1M_10s_video_names
        fake_video_names = ZeroScope_video_names + I2VGEN_XL_video_names + SVD_video_names + \
                           Latte_video_names + OpenSora_video_names + pika_video_names + \
                           SD_video_names + SEINE_video_names + DynamicCrafter_video_names + \
                           VideoCrafter_video_names

        # balance real and fake num
        real_video_names = random.sample(real_video_names, int(self.select_video_nums / 2))
        fake_video_names = random.sample(fake_video_names, int(self.select_video_nums / 2))
        self.video_names = real_video_names + fake_video_names

    def read_txt(self, txt_file_path, label):
        video_names = []
        txt_file_open = open(txt_file_path, mode='r', encoding='utf-8')
        lines = txt_file_open.readlines()
        for line in lines:
            video_name = {}
            video_name['path'] = line.split(' ')[0].strip()
            video_name['label'] = label
            video_names.append(video_name)
        return video_names

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_info = self.video_names[index]
        video_path = video_info['path']
        label = float(video_info['label'])
        pic_names = sorted(os.listdir(video_path))

        aug_list = [albumentations.Resize(224, 224)]

        if random.random() < 0.5:
            aug_list.append(albumentations.HorizontalFlip(p=1.0))
        if random.random() < 0.5:
            quality_score = random.randint(50, 100)
            aug_list.append(albumentations.ImageCompression(quality_lower=quality_score, quality_upper=quality_score))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussNoise(p=1.0))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0))
        if random.random() < 0.001:
            aug_list.append(albumentations.ToGray(p=1.0))

        aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)

        if len(pic_names) >= self.select_frame_nums:
            start_frame = random.randint(0, len(pic_names) - self.select_frame_nums)
            select_frames = pic_names[start_frame:start_frame + self.select_frame_nums]
            frames = []
            for frame_index in range(len(select_frames)):
                pic_name = select_frames[frame_index]
                frame_path = os.path.join(video_path, pic_name)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = trans(image=frame)["image"]
                frames.append(frame.transpose(2, 0, 1)[np.newaxis, :])
        else:
            pad_num = self.select_frame_nums - len(pic_names)
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
        frames = torch.tensor(frames).permute(1, 0, 2, 3)
        frames = torch.tensor(frames)
        return frames, label


class GenVideoVal(Dataset):
    def __init__(self):
        self.select_frame_nums = 8

        # get video txt index file path
        K400_txt_file_path = os.path.join(val_index_file_dir, 'k400.txt')
        Youku_1M_10s_txt_file_path = os.path.join(val_index_file_dir, 'Youku_1M_10s.txt')
        ZeroScope_txt_file_path = os.path.join(val_index_file_dir, 'ZeroScope.txt')
        I2VGEN_XL_txt_file_path = os.path.join(val_index_file_dir, 'I2VGEN_XL.txt')
        SVD_txt_file_path = os.path.join(val_index_file_dir, 'SVD.txt')
        Latte_txt_file_path = os.path.join(val_index_file_dir, 'Latte.txt')
        OpenSora_txt_file_path = os.path.join(val_index_file_dir, 'OpenSora.txt')
        pika_txt_file_path = os.path.join(val_index_file_dir, 'pika.txt')
        SD_txt_file_path = os.path.join(val_index_file_dir, 'SD.txt')
        SEINE_txt_file_path = os.path.join(val_index_file_dir, 'SEINE.txt')
        DynamicCrafter_txt_file_path = os.path.join(val_index_file_dir, 'DynamicCrafter.txt')
        VideoCrafter_txt_file_path = os.path.join(val_index_file_dir, 'VideoCrafter.txt')

        # read video txt index file
        K400_video_names = self.read_txt(K400_txt_file_path, label=0)
        Youku_1M_10s_video_names = self.read_txt(Youku_1M_10s_txt_file_path, label=0)
        ZeroScope_video_names = self.read_txt(ZeroScope_txt_file_path, label=1)
        I2VGEN_XL_video_names = self.read_txt(I2VGEN_XL_txt_file_path, label=1)
        SVD_video_names = self.read_txt(SVD_txt_file_path, label=1)
        Latte_video_names = self.read_txt(Latte_txt_file_path, label=1)
        OpenSora_video_names = self.read_txt(OpenSora_txt_file_path, label=1)
        pika_video_names = self.read_txt(pika_txt_file_path, label=1)
        SD_video_names = self.read_txt(SD_txt_file_path, label=1)
        SEINE_video_names = self.read_txt(SEINE_txt_file_path, label=1)
        DynamicCrafter_video_names = self.read_txt(DynamicCrafter_txt_file_path, label=1)
        VideoCrafter_video_names = self.read_txt(VideoCrafter_txt_file_path, label=1)

        # group real and fake index set
        real_video_names = K400_video_names + Youku_1M_10s_video_names
        fake_video_names = ZeroScope_video_names + I2VGEN_XL_video_names + SVD_video_names + \
                           Latte_video_names + OpenSora_video_names + pika_video_names + \
                           SD_video_names + SEINE_video_names + DynamicCrafter_video_names + \
                           VideoCrafter_video_names
        self.video_names = real_video_names + fake_video_names

    def read_txt(self, txt_file_path, label):
        video_names = []
        txt_file_open = open(txt_file_path, mode='r', encoding='utf-8')
        lines = txt_file_open.readlines()
        for line in lines:
            video_name = {}
            video_name['path'] = line.split(' ')[0].strip()
            video_name['label'] = label
            video_names.append(video_name)
        return video_names

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_info = self.video_names[index]
        video_path = video_info['path']
        label = float(video_info['label'])
        pic_names = sorted(os.listdir(video_path))

        aug_list = [
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
        trans = albumentations.Compose(aug_list)

        if len(pic_names) >= self.select_frame_nums:
            start_frame = random.randint(0, len(pic_names) - self.select_frame_nums)
            select_frames = pic_names[start_frame:start_frame + self.select_frame_nums]
            frames = []
            for frame_index in range(len(select_frames)):
                pic_name = select_frames[frame_index]
                frame_path = os.path.join(video_path, pic_name)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = trans(image=frame)["image"]
                frames.append(frame.transpose(2, 0, 1)[np.newaxis, :])
        else:
            pad_num = self.select_frame_nums - len(pic_names)
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
        frames = torch.tensor(frames).permute(1, 0, 2, 3)
        frames = torch.tensor(frames)
        return frames, label


class GenVideoTest(Dataset):
    def __init__(self):
        self.select_frame_nums = 8

        # get video txt index file path
        MSR_VTT_file_path = os.path.join(test_index_file_dir, 'MSR-VTT.txt')
        Crafter_txt_file_path = os.path.join(test_index_file_dir, 'Crafter.txt')
        Gen2_txt_file_path = os.path.join(test_index_file_dir, 'Gen2.txt')
        HotShot_txt_file_path = os.path.join(test_index_file_dir, 'HotShot.txt')
        Lavie_txt_file_path = os.path.join(test_index_file_dir, 'Lavie.txt')
        ModelScope_txt_file_path = os.path.join(test_index_file_dir, 'ModelScope.txt')
        MoonValley_txt_file_path = os.path.join(test_index_file_dir, 'MoonValley.txt')
        MorphStudio_txt_file_path = os.path.join(test_index_file_dir, 'MorphStudio.txt')
        Show_1_txt_file_path = os.path.join(test_index_file_dir, 'Show_1.txt')
        Sora_txt_file_path = os.path.join(test_index_file_dir, 'Sora.txt')
        WildScrape_txt_file_path = os.path.join(test_index_file_dir, 'WildScrape.txt')

        # read video txt index file
        MSR_VTT_video_names = self.read_txt(MSR_VTT_file_path, label=0)
        Crafter_video_names = self.read_txt(Crafter_txt_file_path, label=0)
        Gen2_video_names = self.read_txt(Gen2_txt_file_path, label=1)
        HotShot_video_names = self.read_txt(HotShot_txt_file_path, label=1)
        Lavie_video_names = self.read_txt(Lavie_txt_file_path, label=1)
        ModelScope_video_names = self.read_txt(ModelScope_txt_file_path, label=1)
        MoonValley_video_names = self.read_txt(MoonValley_txt_file_path, label=1)
        MorphStudio_video_names = self.read_txt(MorphStudio_txt_file_path, label=1)
        Show_1_video_names = self.read_txt(Show_1_txt_file_path, label=1)
        Sora_video_names = self.read_txt(Sora_txt_file_path, label=1)
        WildScrape_video_names = self.read_txt(WildScrape_txt_file_path, label=1)

        # group real and fake index set
        real_video_names = MSR_VTT_video_names
        fake_video_names = Crafter_video_names + Gen2_video_names + HotShot_video_names + \
                           Lavie_video_names + ModelScope_video_names + MoonValley_video_names + \
                           MorphStudio_video_names + Show_1_video_names + Sora_video_names + \
                           WildScrape_video_names
        self.video_names = real_video_names + fake_video_names

    def read_txt(self, txt_file_path, label):
        video_names = []
        txt_file_open = open(txt_file_path, mode='r', encoding='utf-8')
        lines = txt_file_open.readlines()
        for line in lines:
            video_name = {}
            video_name['path'] = line.split(' ')[0].strip()
            video_name['label'] = label
            video_names.append(video_name)
        return video_names

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_info = self.video_names[index]
        video_path = video_info['path']
        label = float(video_info['label'])
        pic_names = sorted(os.listdir(video_path))

        aug_list = [
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
        trans = albumentations.Compose(aug_list)

        if len(pic_names) >= self.select_frame_nums:
            start_frame = random.randint(0, len(pic_names) - self.select_frame_nums)
            select_frames = pic_names[start_frame:start_frame + self.select_frame_nums]
            frames = []
            for frame_index in range(len(select_frames)):
                pic_name = select_frames[frame_index]
                frame_path = os.path.join(video_path, pic_name)
                frame = cv2.imread(frame_path)
                channels = frame.shape[2] if len(frame.shape) == 3 else 1
                if channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = trans(image=frame)["image"]
                frames.append(frame.transpose(2, 0, 1)[np.newaxis, :])
        else:
            pad_num = self.select_frame_nums - len(pic_names)
            frames = []
            for frame_index in range(len(pic_names)):
                pic_name = pic_names[frame_index]
                frame_path = os.path.join(video_path, pic_name)
                frame = cv2.imread(frame_path)
                channels = frame.shape[2] if len(frame.shape) == 3 else 1
                if channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = trans(image=frame)["image"]
                frames.append(frame.transpose(2, 0, 1)[np.newaxis, :])
            for i in range(pad_num):
                frames.append(np.zeros(shape=(1, 3, 224, 224)))

        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames).permute(1, 0, 2, 3)
        frames = torch.tensor(frames)
        return frames, label
