import os
import random
import numpy as np
import torch

from xclip_tools import init_xclip_model, get_videos_xclip_feats
from vit_tools import init_vit_model, get_videos_vit_feats
from clip_tools import init_clip_model, get_videos_clip_feats
from viclip_tools import init_viclip_model, get_videos_viclip_feats
from videomae_tools import init_videomae_model, get_videos_videomae_feats


def random_get_video_paths(directory):
    video_files = []
    video_names = os.listdir(directory)
    for video_name in video_names:
        video_files.append(os.path.join(directory, video_name))
    selected_files = random.sample(video_files, 200)
    return selected_files


def save_xclip_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
):
    device = torch.device('cuda:0')
    model, trans = init_xclip_model(device)
    DynamicCrafter_video_feats = get_videos_xclip_feats(DynamicCrafter_video_paths, model, trans, device)
    print('DynamicCrafter_video_feats')
    I2VGEN_XL_video_feats = get_videos_xclip_feats(I2VGEN_XL_video_paths, model, trans, device)
    print('I2VGEN_XL_video_feats')
    Latte_video_feats = get_videos_xclip_feats(Latte_video_paths, model, trans, device)
    print('Latte_video_feats')
    OpenSora_video_feats = get_videos_xclip_feats(OpenSora_video_paths, model, trans, device)
    print('OpenSora_video_feats')
    pika_video_feats = get_videos_xclip_feats(pika_video_paths, model, trans, device)
    print('pika_video_feats')
    SD_video_feats = get_videos_xclip_feats(SD_video_paths, model, trans, device)
    print('SD_video_feats')
    SEINE_video_feats = get_videos_xclip_feats(SEINE_video_paths, model, trans, device)
    print('SEINE_video_feats')
    SVD_video_feats = get_videos_xclip_feats(SVD_video_paths, model, trans, device)
    print('SVD_video_feats')
    VideoCrafter_video_feats = get_videos_xclip_feats(VideoCrafter_video_paths, model, trans, device)
    print('VideoCrafter_video_feats')
    ZeroScope_video_feats = get_videos_xclip_feats(ZeroScope_video_paths, model, trans, device)
    print('ZeroScope_video_feats')
    Youku_video_feats = get_videos_xclip_feats(Youku_video_paths, model, trans, device)
    print('Youku_video_feats')
    K400_video_feats = get_videos_xclip_feats(K400_video_paths, model, trans, device)
    print('K400_video_feats')
    Crafter_video_feats = get_videos_xclip_feats(Crafter_video_paths, model, trans, device)
    print('Crafter_video_feats')
    Gen2_video_feats = get_videos_xclip_feats(Gen2_video_paths, model, trans, device)
    print('Gen2_video_feats')
    HotShot_video_feats = get_videos_xclip_feats(HotShot_video_paths, model, trans, device)
    print('HotShot_video_feats')
    Lavie_video_feats = get_videos_xclip_feats(Lavie_video_paths, model, trans, device)
    print('Lavie_video_feats')
    ModelScope_video_feats = get_videos_xclip_feats(ModelScope_video_paths, model, trans, device)
    print('ModelScope_video_feats')
    MoonValley_video_feats = get_videos_xclip_feats(MoonValley_video_paths, model, trans, device)
    print('MoonValley_video_feats')
    MorphStudio_video_feats = get_videos_xclip_feats(MorphStudio_video_paths, model, trans, device)
    print('MorphStudio_video_feats')
    Show_1_video_feats = get_videos_xclip_feats(Show_1_video_paths, model, trans, device)
    print('Show_1_video_feats')

    feats = [
        DynamicCrafter_video_feats,
        I2VGEN_XL_video_feats,
        Latte_video_feats,
        OpenSora_video_feats,
        pika_video_feats,
        SD_video_feats,
        SEINE_video_feats,
        SVD_video_feats,
        VideoCrafter_video_feats,
        ZeroScope_video_feats,
        Youku_video_feats,
        K400_video_feats,
        Crafter_video_feats,
        Gen2_video_feats,
        HotShot_video_feats,
        Lavie_video_feats,
        ModelScope_video_feats,
        MoonValley_video_feats,
        MorphStudio_video_feats,
        Show_1_video_feats
    ]

    feats = np.vstack(feats)
    print(feats.shape)
    np.save("xclip_feats.npy", feats)


def save_vit_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
):
    device = torch.device('cuda:0')
    model, preprocess = init_vit_model(device)
    DynamicCrafter_video_feats = get_videos_vit_feats(DynamicCrafter_video_paths, model, preprocess, device)
    print('DynamicCrafter_video_feats')
    I2VGEN_XL_video_feats = get_videos_vit_feats(I2VGEN_XL_video_paths, model, preprocess, device)
    print('I2VGEN_XL_video_feats')
    Latte_video_feats = get_videos_vit_feats(Latte_video_paths, model, preprocess, device)
    print('Latte_video_feats')
    OpenSora_video_feats = get_videos_vit_feats(OpenSora_video_paths, model, preprocess, device)
    print('OpenSora_video_feats')
    pika_video_feats = get_videos_vit_feats(pika_video_paths, model, preprocess, device)
    print('pika_video_feats')
    SD_video_feats = get_videos_vit_feats(SD_video_paths, model, preprocess, device)
    print('SD_video_feats')
    SEINE_video_feats = get_videos_vit_feats(SEINE_video_paths, model, preprocess, device)
    print('SEINE_video_feats')
    SVD_video_feats = get_videos_vit_feats(SVD_video_paths, model, preprocess, device)
    print('SVD_video_feats')
    VideoCrafter_video_feats = get_videos_vit_feats(VideoCrafter_video_paths, model, preprocess, device)
    print('VideoCrafter_video_feats')
    ZeroScope_video_feats = get_videos_vit_feats(ZeroScope_video_paths, model, preprocess, device)
    print('ZeroScope_video_feats')
    Youku_video_feats = get_videos_vit_feats(Youku_video_paths, model, preprocess, device)
    print('Youku_video_feats')
    K400_video_feats = get_videos_vit_feats(K400_video_paths, model, preprocess, device)
    print('K400_video_feats')
    Crafter_video_feats = get_videos_vit_feats(Crafter_video_paths, model, preprocess, device)
    print('Crafter_video_feats')
    Gen2_video_feats = get_videos_vit_feats(Gen2_video_paths, model, preprocess, device)
    print('Gen2_video_feats')
    HotShot_video_feats = get_videos_vit_feats(HotShot_video_paths, model, preprocess, device)
    print('HotShot_video_feats')
    Lavie_video_feats = get_videos_vit_feats(Lavie_video_paths, model, preprocess, device)
    print('Lavie_video_feats')
    ModelScope_video_feats = get_videos_vit_feats(ModelScope_video_paths, model, preprocess, device)
    print('ModelScope_video_feats')
    MoonValley_video_feats = get_videos_vit_feats(MoonValley_video_paths, model, preprocess, device)
    print('MoonValley_video_feats')
    MorphStudio_video_feats = get_videos_vit_feats(MorphStudio_video_paths, model, preprocess, device)
    print('MorphStudio_video_feats')
    Show_1_video_feats = get_videos_vit_feats(Show_1_video_paths, model, preprocess, device)
    print('Show_1_video_feats')

    feats = [
        DynamicCrafter_video_feats,
        I2VGEN_XL_video_feats,
        Latte_video_feats,
        OpenSora_video_feats,
        pika_video_feats,
        SD_video_feats,
        SEINE_video_feats,
        SVD_video_feats,
        VideoCrafter_video_feats,
        ZeroScope_video_feats,
        Youku_video_feats,
        K400_video_feats,
        Crafter_video_feats,
        Gen2_video_feats,
        HotShot_video_feats,
        Lavie_video_feats,
        ModelScope_video_feats,
        MoonValley_video_feats,
        MorphStudio_video_feats,
        Show_1_video_feats
    ]

    feats = np.vstack(feats)
    print(feats.shape)
    np.save("vit_feats.npy", feats)


def save_clip_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
):
    device = torch.device('cuda:0')
    model, preprocess = init_clip_model(device)
    DynamicCrafter_video_feats = get_videos_clip_feats(DynamicCrafter_video_paths, model, preprocess, device)
    print('DynamicCrafter_video_feats')
    I2VGEN_XL_video_feats = get_videos_clip_feats(I2VGEN_XL_video_paths, model, preprocess, device)
    print('I2VGEN_XL_video_feats')
    Latte_video_feats = get_videos_clip_feats(Latte_video_paths, model, preprocess, device)
    print('Latte_video_feats')
    OpenSora_video_feats = get_videos_clip_feats(OpenSora_video_paths, model, preprocess, device)
    print('OpenSora_video_feats')
    pika_video_feats = get_videos_clip_feats(pika_video_paths, model, preprocess, device)
    print('pika_video_feats')
    SD_video_feats = get_videos_clip_feats(SD_video_paths, model, preprocess, device)
    print('SD_video_feats')
    SEINE_video_feats = get_videos_clip_feats(SEINE_video_paths, model, preprocess, device)
    print('SEINE_video_feats')
    SVD_video_feats = get_videos_clip_feats(SVD_video_paths, model, preprocess, device)
    print('SVD_video_feats')
    VideoCrafter_video_feats = get_videos_clip_feats(VideoCrafter_video_paths, model, preprocess, device)
    print('VideoCrafter_video_feats')
    ZeroScope_video_feats = get_videos_clip_feats(ZeroScope_video_paths, model, preprocess, device)
    print('ZeroScope_video_feats')
    Youku_video_feats = get_videos_clip_feats(Youku_video_paths, model, preprocess, device)
    print('Youku_video_feats')
    K400_video_feats = get_videos_clip_feats(K400_video_paths, model, preprocess, device)
    print('K400_video_feats')
    Crafter_video_feats = get_videos_clip_feats(Crafter_video_paths, model, preprocess, device)
    print('Crafter_video_feats')
    Gen2_video_feats = get_videos_clip_feats(Gen2_video_paths, model, preprocess, device)
    print('Gen2_video_feats')
    HotShot_video_feats = get_videos_clip_feats(HotShot_video_paths, model, preprocess, device)
    print('HotShot_video_feats')
    Lavie_video_feats = get_videos_clip_feats(Lavie_video_paths, model, preprocess, device)
    print('Lavie_video_feats')
    ModelScope_video_feats = get_videos_clip_feats(ModelScope_video_paths, model, preprocess, device)
    print('ModelScope_video_feats')
    MoonValley_video_feats = get_videos_clip_feats(MoonValley_video_paths, model, preprocess, device)
    print('MoonValley_video_feats')
    MorphStudio_video_feats = get_videos_clip_feats(MorphStudio_video_paths, model, preprocess, device)
    print('MorphStudio_video_feats')
    Show_1_video_feats = get_videos_clip_feats(Show_1_video_paths, model, preprocess, device)
    print('Show_1_video_feats')

    feats = [
        DynamicCrafter_video_feats,
        I2VGEN_XL_video_feats,
        Latte_video_feats,
        OpenSora_video_feats,
        pika_video_feats,
        SD_video_feats,
        SEINE_video_feats,
        SVD_video_feats,
        VideoCrafter_video_feats,
        ZeroScope_video_feats,
        Youku_video_feats,
        K400_video_feats,
        Crafter_video_feats,
        Gen2_video_feats,
        HotShot_video_feats,
        Lavie_video_feats,
        ModelScope_video_feats,
        MoonValley_video_feats,
        MorphStudio_video_feats,
        Show_1_video_feats
    ]

    feats = np.vstack(feats)
    print(feats.shape)
    np.save("clip_feats.npy", feats)


def save_viclip_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
):
    device = torch.device('cuda:0')
    model = init_viclip_model(device)
    DynamicCrafter_video_feats = get_videos_viclip_feats(DynamicCrafter_video_paths, model, device)
    print('DynamicCrafter_video_feats')
    I2VGEN_XL_video_feats = get_videos_viclip_feats(I2VGEN_XL_video_paths, model, device)
    print('I2VGEN_XL_video_feats')
    Latte_video_feats = get_videos_viclip_feats(Latte_video_paths, model, device)
    print('Latte_video_feats')
    OpenSora_video_feats = get_videos_viclip_feats(OpenSora_video_paths, model, device)
    print('OpenSora_video_feats')
    pika_video_feats = get_videos_viclip_feats(pika_video_paths, model, device)
    print('pika_video_feats')
    SD_video_feats = get_videos_viclip_feats(SD_video_paths, model, device)
    print('SD_video_feats')
    SEINE_video_feats = get_videos_viclip_feats(SEINE_video_paths, model, device)
    print('SEINE_video_feats')
    SVD_video_feats = get_videos_viclip_feats(SVD_video_paths, model, device)
    print('SVD_video_feats')
    VideoCrafter_video_feats = get_videos_viclip_feats(VideoCrafter_video_paths, model, device)
    print('VideoCrafter_video_feats')
    ZeroScope_video_feats = get_videos_viclip_feats(ZeroScope_video_paths, model, device)
    print('ZeroScope_video_feats')
    Youku_video_feats = get_videos_viclip_feats(Youku_video_paths, model, device)
    print('Youku_video_feats')
    K400_video_feats = get_videos_viclip_feats(K400_video_paths, model, device)
    print('K400_video_feats')
    Crafter_video_feats = get_videos_viclip_feats(Crafter_video_paths, model, device)
    print('Crafter_video_feats')
    Gen2_video_feats = get_videos_viclip_feats(Gen2_video_paths, model, device)
    print('Gen2_video_feats')
    HotShot_video_feats = get_videos_viclip_feats(HotShot_video_paths, model, device)
    print('HotShot_video_feats')
    Lavie_video_feats = get_videos_viclip_feats(Lavie_video_paths, model, device)
    print('Lavie_video_feats')
    ModelScope_video_feats = get_videos_viclip_feats(ModelScope_video_paths, model, device)
    print('ModelScope_video_feats')
    MoonValley_video_feats = get_videos_viclip_feats(MoonValley_video_paths, model, device)
    print('MoonValley_video_feats')
    MorphStudio_video_feats = get_videos_viclip_feats(MorphStudio_video_paths, model, device)
    print('MorphStudio_video_feats')
    Show_1_video_feats = get_videos_viclip_feats(Show_1_video_paths, model, device)
    print('Show_1_video_feats')

    feats = [
        DynamicCrafter_video_feats,
        I2VGEN_XL_video_feats,
        Latte_video_feats,
        OpenSora_video_feats,
        pika_video_feats,
        SD_video_feats,
        SEINE_video_feats,
        SVD_video_feats,
        VideoCrafter_video_feats,
        ZeroScope_video_feats,
        Youku_video_feats,
        K400_video_feats,
        Crafter_video_feats,
        Gen2_video_feats,
        HotShot_video_feats,
        Lavie_video_feats,
        ModelScope_video_feats,
        MoonValley_video_feats,
        MorphStudio_video_feats,
        Show_1_video_feats
    ]

    feats = np.vstack(feats)
    print(feats.shape)
    np.save("viclip_feats.npy", feats)


def save_videomae_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
):
    device = torch.device('cuda:0')
    model = init_videomae_model(device)
    DynamicCrafter_video_feats = get_videos_videomae_feats(DynamicCrafter_video_paths, model, device)
    print('DynamicCrafter_video_feats')
    I2VGEN_XL_video_feats = get_videos_videomae_feats(I2VGEN_XL_video_paths, model, device)
    print('I2VGEN_XL_video_feats')
    Latte_video_feats = get_videos_videomae_feats(Latte_video_paths, model, device)
    print('Latte_video_feats')
    OpenSora_video_feats = get_videos_videomae_feats(OpenSora_video_paths, model, device)
    print('OpenSora_video_feats')
    pika_video_feats = get_videos_videomae_feats(pika_video_paths, model, device)
    print('pika_video_feats')
    SD_video_feats = get_videos_videomae_feats(SD_video_paths, model, device)
    print('SD_video_feats')
    SEINE_video_feats = get_videos_videomae_feats(SEINE_video_paths, model, device)
    print('SEINE_video_feats')
    SVD_video_feats = get_videos_videomae_feats(SVD_video_paths, model, device)
    print('SVD_video_feats')
    VideoCrafter_video_feats = get_videos_videomae_feats(VideoCrafter_video_paths, model, device)
    print('VideoCrafter_video_feats')
    ZeroScope_video_feats = get_videos_videomae_feats(ZeroScope_video_paths, model, device)
    print('ZeroScope_video_feats')
    Youku_video_feats = get_videos_videomae_feats(Youku_video_paths, model, device)
    print('Youku_video_feats')
    K400_video_feats = get_videos_videomae_feats(K400_video_paths, model, device)
    print('K400_video_feats')
    Crafter_video_feats = get_videos_videomae_feats(Crafter_video_paths, model, device)
    print('Crafter_video_feats')
    Gen2_video_feats = get_videos_videomae_feats(Gen2_video_paths, model, device)
    print('Gen2_video_feats')
    HotShot_video_feats = get_videos_videomae_feats(HotShot_video_paths, model, device)
    print('HotShot_video_feats')
    Lavie_video_feats = get_videos_videomae_feats(Lavie_video_paths, model, device)
    print('Lavie_video_feats')
    ModelScope_video_feats = get_videos_videomae_feats(ModelScope_video_paths, model, device)
    print('ModelScope_video_feats')
    MoonValley_video_feats = get_videos_videomae_feats(MoonValley_video_paths, model, device)
    print('MoonValley_video_feats')
    MorphStudio_video_feats = get_videos_videomae_feats(MorphStudio_video_paths, model, device)
    print('MorphStudio_video_feats')
    Show_1_video_feats = get_videos_videomae_feats(Show_1_video_paths, model, device)
    print('Show_1_video_feats')

    feats = [
        DynamicCrafter_video_feats,
        I2VGEN_XL_video_feats,
        Latte_video_feats,
        OpenSora_video_feats,
        pika_video_feats,
        SD_video_feats,
        SEINE_video_feats,
        SVD_video_feats,
        VideoCrafter_video_feats,
        ZeroScope_video_feats,
        Youku_video_feats,
        K400_video_feats,
        Crafter_video_feats,
        Gen2_video_feats,
        HotShot_video_feats,
        Lavie_video_feats,
        ModelScope_video_feats,
        MoonValley_video_feats,
        MorphStudio_video_feats,
        Show_1_video_feats
    ]

    feats = np.vstack(feats)
    print(feats.shape)
    np.save("videomae_feats.npy", feats)


def main():
    video_dir = '/data2/ljc/dataset/GenVideo_frame/'
    DynamicCrafter_video_dir = os.path.join(video_dir, 'train_DynamicCrafter')
    I2VGEN_XL_video_dir = os.path.join(video_dir, 'train_I2VGEN_XL')
    Latte_video_dir = os.path.join(video_dir, 'train_Latte')
    OpenSora_video_dir = os.path.join(video_dir, 'train_OpenSora')
    pika_video_dir = os.path.join(video_dir, 'train_pika')
    SD_video_dir = os.path.join(video_dir, 'train_SD')
    SEINE_video_dir = os.path.join(video_dir, 'train_SEINE')
    SVD_video_dir = os.path.join(video_dir, 'train_SVD')
    VideoCrafter_video_dir = os.path.join(video_dir, 'train_VideoCrafter')
    ZeroScope_video_dir = os.path.join(video_dir, 'train_ZeroScope')
    Youku_video_dir = os.path.join(video_dir, 'Youku_1M_10s', '0000000_0009999')
    K400_video_dir = os.path.join(video_dir, 'k400', 'train')
    Crafter_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'Crafter')
    Gen2_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'Gen2')
    HotShot_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'HotShot')
    Lavie_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'Lavie')
    ModelScope_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'ModelScope')
    MoonValley_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'MoonValley')
    MorphStudio_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'MorphStudio')
    Show_1_video_dir = os.path.join(video_dir, 'GenVideo-Val', 'Fake', 'MorphStudio')

    DynamicCrafter_video_paths = random_get_video_paths(DynamicCrafter_video_dir)
    I2VGEN_XL_video_paths = random_get_video_paths(I2VGEN_XL_video_dir)
    Latte_video_paths = random_get_video_paths(Latte_video_dir)
    OpenSora_video_paths = random_get_video_paths(OpenSora_video_dir)
    pika_video_paths = random_get_video_paths(pika_video_dir)
    SD_video_paths = random_get_video_paths(SD_video_dir)
    SEINE_video_paths = random_get_video_paths(SEINE_video_dir)
    SVD_video_paths = random_get_video_paths(SVD_video_dir)
    VideoCrafter_video_paths = random_get_video_paths(VideoCrafter_video_dir)
    ZeroScope_video_paths = random_get_video_paths(ZeroScope_video_dir)
    Youku_video_paths = random_get_video_paths(Youku_video_dir)
    K400_video_paths = random_get_video_paths(K400_video_dir)
    Crafter_video_paths = random_get_video_paths(Crafter_video_dir)
    Gen2_video_paths = random_get_video_paths(Gen2_video_dir)
    HotShot_video_paths = random_get_video_paths(HotShot_video_dir)
    Lavie_video_paths = random_get_video_paths(Lavie_video_dir)
    ModelScope_video_paths = random_get_video_paths(ModelScope_video_dir)
    MoonValley_video_paths = random_get_video_paths(MoonValley_video_dir)
    MorphStudio_video_paths = random_get_video_paths(MorphStudio_video_dir)
    Show_1_video_paths = random_get_video_paths(Show_1_video_dir)

    save_xclip_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
    )

    save_vit_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
    )

    save_clip_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
    )

    save_viclip_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
    )

    save_videomae_feats(
        DynamicCrafter_video_paths,
        I2VGEN_XL_video_paths,
        Latte_video_paths,
        OpenSora_video_paths,
        pika_video_paths,
        SD_video_paths,
        SEINE_video_paths,
        SVD_video_paths,
        VideoCrafter_video_paths,
        ZeroScope_video_paths,
        Youku_video_paths,
        K400_video_paths,
        Crafter_video_paths,
        Gen2_video_paths,
        HotShot_video_paths,
        Lavie_video_paths,
        ModelScope_video_paths,
        MoonValley_video_paths,
        MorphStudio_video_paths,
        Show_1_video_paths,
    )


if __name__ == '__main__':
    main()
