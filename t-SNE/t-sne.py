import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def t_sne_1(feats1, feats2, feats3, feats4, name1, name2, name):
    data = np.vstack([feats1, feats2, feats3, feats4])
    labels = np.concatenate([np.full(200, i) for i in range(4)])

    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        learning_rate=200,
        random_state=42
    )
    data_2d = tsne.fit_transform(data_normalized)

    plt.figure(figsize=(12, 9), dpi=100)

    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
    markers = ['o', 'o', 'o', 'o']

    classes = ['Youku', 'K400', name1, name2]
    for i in range(4):
        plt.scatter(
            data_2d[labels == i, 0],
            data_2d[labels == i, 1],
            c=colors[i],
            marker=markers[i],
            s=50,
            edgecolor='w',
            linewidth=0.5,
            alpha=0.8,
            label=classes[i]
        )

    plt.legend(fontsize=12, loc='best')
    plt.title(f'{name} t-SNE Visualization of 800 Samples (512D → 2D)', fontsize=14, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    name = 'D:/code/Text2Video/Detec/Code/ViCLIP/pic/ALL4/' + name + '_' + name1 + '_' + name2 + '.png'
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()


def t_sne_2(feats1, feats2, feats3, feats4, name1, name2, name):
    data = np.vstack([feats1, feats2, feats3, feats4])
    labels = np.concatenate([np.full(200, i) for i in range(4)])

    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    tsne = TSNE(
        n_components=2,
        perplexity=40,
        n_iter=1500,
        learning_rate=300,
        random_state=42
    )
    data_2d = tsne.fit_transform(data_normalized)

    plt.figure(figsize=(12, 9), dpi=100)

    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
    markers = ['o', 'o', 'o', 'o']
    classes = ['Youku', 'K400', name1, name2]

    for i in range(4):
        plt.scatter(
            data_2d[labels == i, 0],
            data_2d[labels == i, 1],
            c=colors[i],
            marker=markers[i],
            s=50,
            edgecolor='w',
            linewidth=0.5,
            alpha=0.8,
            label=classes[i]
        )

    plt.legend(fontsize=12, loc='best')
    plt.title(f'{name} t-SNE Visualization of 800 Samples (768D → 2D)', fontsize=14, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    # 优化坐标轴显示
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    name = 'D:/code/Text2Video/Detec/Code/ViCLIP/pic/ALL4/' + name + '_' + name1 + '_' + name2 + '.png'
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()


def clip_feats_main():
    feats = np.load("./feats2/clip_feats.npy")
    print('./feats2/clip_feats :', feats.shape)
    DynamicCrafter_feats = feats[0:200, :]
    I2VGEN_XL_feats = feats[200:400, :]
    Latte_feats = feats[400:600, :]
    OpenSora_feats = feats[600:800, :]
    pika_feats = feats[800:1000, :]
    SD_feats = feats[1000:1200, :]
    SEINE_feats = feats[1200:1400, :]
    SVD_feats = feats[1400:1600, :]
    VideoCrafter_feats = feats[1600:1800, :]
    ZeroScope_feats = feats[1800:2000, :]
    Youku_feats = feats[2000:2200, :]
    K400_feats = feats[2200:2400, :]
    Crafter_feats = feats[2400:2600, :]
    Gen2_feats = feats[2600:2800, :]
    HotShot_feats = feats[2800:3000, :]
    Lavie_feats = feats[3000:3200, :]
    ModelScope_feats = feats[3200:3400, :]
    MoonValley_feats = feats[3400:3600, :]
    MorphStudio_feats = feats[3600:3800, :]
    Show_1_feats = feats[3800:4000, :]

    # t_sne_1(Youku_feats, K400_feats, DynamicCrafter_feats, I2VGEN_XL_feats, 'DynamicCrafter', 'I2VGEN_XL', 'clip')
    # t_sne_1(Youku_feats, K400_feats, Latte_feats, OpenSora_feats, 'Latte', 'OpenSora', 'clip')
    # t_sne_1(Youku_feats, K400_feats, pika_feats, SD_feats, 'pika', 'SD', 'clip')
    t_sne_1(Youku_feats, K400_feats, SEINE_feats, SVD_feats, 'SEINE', 'SVD', 'clip')
    # t_sne_1(Youku_feats, K400_feats, VideoCrafter_feats, ZeroScope_feats, 'VideoCrafter', 'ZeroScope', 'clip')
    # t_sne_1(Youku_feats, K400_feats, Crafter_feats, Gen2_feats, 'Crafter', 'Gen2', 'clip')
    # t_sne_1(Youku_feats, K400_feats, HotShot_feats, Lavie_feats, 'HotShot', 'Lavie', 'clip')
    # t_sne_1(Youku_feats, K400_feats, ModelScope_feats, MoonValley_feats, 'ModelScope', 'MoonValley', 'clip')
    # t_sne_1(Youku_feats, K400_feats, MorphStudio_feats, Show_1_feats, 'MorphStudio', 'Show_1', 'clip')


def viclip_feats_main():
    feats = np.load("./feats2/viclip_feats.npy")
    print('./feats2/viclip_feats :', feats.shape)
    DynamicCrafter_feats = feats[0:200, :]
    I2VGEN_XL_feats = feats[200:400, :]
    Latte_feats = feats[400:600, :]
    OpenSora_feats = feats[600:800, :]
    pika_feats = feats[800:1000, :]
    SD_feats = feats[1000:1200, :]
    SEINE_feats = feats[1200:1400, :]
    SVD_feats = feats[1400:1600, :]
    VideoCrafter_feats = feats[1600:1800, :]
    ZeroScope_feats = feats[1800:2000, :]
    Youku_feats = feats[2000:2200, :]
    K400_feats = feats[2200:2400, :]
    Crafter_feats = feats[2400:2600, :]
    Gen2_feats = feats[2600:2800, :]
    HotShot_feats = feats[2800:3000, :]
    Lavie_feats = feats[3000:3200, :]
    ModelScope_feats = feats[3200:3400, :]
    MoonValley_feats = feats[3400:3600, :]
    MorphStudio_feats = feats[3600:3800, :]
    Show_1_feats = feats[3800:4000, :]

    # t_sne_2(Youku_feats, K400_feats, DynamicCrafter_feats, I2VGEN_XL_feats, 'DynamicCrafter', 'I2VGEN_XL', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, Latte_feats, OpenSora_feats, 'Latte', 'OpenSora', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, pika_feats, SD_feats, 'pika', 'SD', 'viclip')
    t_sne_2(Youku_feats, K400_feats, SEINE_feats, SVD_feats, 'SEINE', 'SVD', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, VideoCrafter_feats, ZeroScope_feats, 'VideoCrafter', 'ZeroScope', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, Crafter_feats, Gen2_feats, 'Crafter', 'Gen2', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, HotShot_feats, Lavie_feats, 'HotShot', 'Lavie', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, ModelScope_feats, MoonValley_feats, 'ModelScope', 'MoonValley', 'viclip')
    # t_sne_2(Youku_feats, K400_feats, MorphStudio_feats, Show_1_feats, 'MorphStudio', 'Show_1', 'viclip')


def vit_feats_main():
    feats = np.load("./feats2/vit_feats.npy")
    print('./feats2/vit_feats :', feats.shape)
    DynamicCrafter_feats = feats[0:200, :]
    I2VGEN_XL_feats = feats[200:400, :]
    Latte_feats = feats[400:600, :]
    OpenSora_feats = feats[600:800, :]
    pika_feats = feats[800:1000, :]
    SD_feats = feats[1000:1200, :]
    SEINE_feats = feats[1200:1400, :]
    SVD_feats = feats[1400:1600, :]
    VideoCrafter_feats = feats[1600:1800, :]
    ZeroScope_feats = feats[1800:2000, :]
    Youku_feats = feats[2000:2200, :]
    K400_feats = feats[2200:2400, :]
    Crafter_feats = feats[2400:2600, :]
    Gen2_feats = feats[2600:2800, :]
    HotShot_feats = feats[2800:3000, :]
    Lavie_feats = feats[3000:3200, :]
    ModelScope_feats = feats[3200:3400, :]
    MoonValley_feats = feats[3400:3600, :]
    MorphStudio_feats = feats[3600:3800, :]
    Show_1_feats = feats[3800:4000, :]

    # t_sne_2(Youku_feats, K400_feats, DynamicCrafter_feats, I2VGEN_XL_feats, 'DynamicCrafter', 'I2VGEN_XL', 'vit')
    # t_sne_2(Youku_feats, K400_feats, Latte_feats, OpenSora_feats, 'Latte', 'OpenSora', 'vit')
    # t_sne_2(Youku_feats, K400_feats, pika_feats, SD_feats, 'pika', 'SD', 'vit')
    t_sne_2(Youku_feats, K400_feats, SEINE_feats, SVD_feats, 'SEINE', 'SVD', 'vit')
    # t_sne_2(Youku_feats, K400_feats, VideoCrafter_feats, ZeroScope_feats, 'VideoCrafter', 'ZeroScope', 'vit')
    # t_sne_2(Youku_feats, K400_feats, Crafter_feats, Gen2_feats, 'Crafter', 'Gen2', 'vit')
    # t_sne_2(Youku_feats, K400_feats, HotShot_feats, Lavie_feats, 'HotShot', 'Lavie', 'vit')
    # t_sne_2(Youku_feats, K400_feats, ModelScope_feats, MoonValley_feats, 'ModelScope', 'MoonValley', 'vit')
    # t_sne_2(Youku_feats, K400_feats, MorphStudio_feats, Show_1_feats, 'MorphStudio', 'Show_1', 'vit')


def xclip_feats_main():
    feats = np.load("./feats2/xclip_feats.npy")
    print('./feats2/xclip_feats :', feats.shape)
    DynamicCrafter_feats = feats[0:200, :]
    I2VGEN_XL_feats = feats[200:400, :]
    Latte_feats = feats[400:600, :]
    OpenSora_feats = feats[600:800, :]
    pika_feats = feats[800:1000, :]
    SD_feats = feats[1000:1200, :]
    SEINE_feats = feats[1200:1400, :]
    SVD_feats = feats[1400:1600, :]
    VideoCrafter_feats = feats[1600:1800, :]
    ZeroScope_feats = feats[1800:2000, :]
    Youku_feats = feats[2000:2200, :]
    K400_feats = feats[2200:2400, :]
    Crafter_feats = feats[2400:2600, :]
    Gen2_feats = feats[2600:2800, :]
    HotShot_feats = feats[2800:3000, :]
    Lavie_feats = feats[3000:3200, :]
    ModelScope_feats = feats[3200:3400, :]
    MoonValley_feats = feats[3400:3600, :]
    MorphStudio_feats = feats[3600:3800, :]
    Show_1_feats = feats[3800:4000, :]

    # t_sne_2(Youku_feats, K400_feats, DynamicCrafter_feats, I2VGEN_XL_feats, 'DynamicCrafter', 'I2VGEN_XL', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, Latte_feats, OpenSora_feats, 'Latte', 'OpenSora', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, pika_feats, SD_feats, 'pika', 'SD', 'xclip')
    t_sne_2(Youku_feats, K400_feats, SEINE_feats, SVD_feats, 'SEINE', 'SVD', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, VideoCrafter_feats, ZeroScope_feats, 'VideoCrafter', 'ZeroScope', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, Crafter_feats, Gen2_feats, 'Crafter', 'Gen2', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, HotShot_feats, Lavie_feats, 'HotShot', 'Lavie', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, ModelScope_feats, MoonValley_feats, 'ModelScope', 'MoonValley', 'xclip')
    # t_sne_2(Youku_feats, K400_feats, MorphStudio_feats, Show_1_feats, 'MorphStudio', 'Show_1', 'xclip')


def videomae_feats_main():
    feats = np.load("./feats2/videomae_feats.npy")
    print('./feats2/videomae_feats :', feats.shape)
    DynamicCrafter_feats = feats[0:200, :]
    I2VGEN_XL_feats = feats[200:400, :]
    Latte_feats = feats[400:600, :]
    OpenSora_feats = feats[600:800, :]
    pika_feats = feats[800:1000, :]
    SD_feats = feats[1000:1200, :]
    SEINE_feats = feats[1200:1400, :]
    SVD_feats = feats[1400:1600, :]
    VideoCrafter_feats = feats[1600:1800, :]
    ZeroScope_feats = feats[1800:2000, :]
    Youku_feats = feats[2000:2200, :]
    K400_feats = feats[2200:2400, :]
    Crafter_feats = feats[2400:2600, :]
    Gen2_feats = feats[2600:2800, :]
    HotShot_feats = feats[2800:3000, :]
    Lavie_feats = feats[3000:3200, :]
    ModelScope_feats = feats[3200:3400, :]
    MoonValley_feats = feats[3400:3600, :]
    MorphStudio_feats = feats[3600:3800, :]
    Show_1_feats = feats[3800:4000, :]

    # t_sne_2(Youku_feats, K400_feats, DynamicCrafter_feats, I2VGEN_XL_feats, 'DynamicCrafter', 'I2VGEN_XL', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, Latte_feats, OpenSora_feats, 'Latte', 'OpenSora', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, pika_feats, SD_feats, 'pika', 'SD', 'videomae')
    t_sne_2(Youku_feats, K400_feats, SEINE_feats, SVD_feats, 'SEINE', 'SVD', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, VideoCrafter_feats, ZeroScope_feats, 'VideoCrafter', 'ZeroScope', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, Crafter_feats, Gen2_feats, 'Crafter', 'Gen2', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, HotShot_feats, Lavie_feats, 'HotShot', 'Lavie', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, ModelScope_feats, MoonValley_feats, 'ModelScope', 'MoonValley', 'videomae')
    # t_sne_2(Youku_feats, K400_feats, MorphStudio_feats, Show_1_feats, 'MorphStudio', 'Show_1', 'videomae')


if __name__ == '__main__':
    clip_feats_main()
    viclip_feats_main()
    vit_feats_main()
    xclip_feats_main()
    videomae_feats_main()
