import os
import random


def main(index_file_path, save_train_path, save_val_path):
    index_file_open = open(index_file_path, mode='r', encoding='utf-8')
    lines = index_file_open.readlines()
    index_file_open.close()

    total_num = len(lines)
    split_size = int(0.1 * total_num)
    random.shuffle(lines)
    val_lines = lines[:split_size]
    train_lines = lines[split_size:]

    save_train_open = open(save_train_path, mode='w', encoding='utf-8')
    for line in train_lines:
        save_train_open.write(line)
    save_train_open.close()

    save_val_open = open(save_val_path, mode='w', encoding='utf-8')
    for line in val_lines:
        save_val_open.write(line)
    save_val_open.close()


if __name__ == '__main__':

    # Modify the following path
    index_file_dir = '/data2/ljc/dataset/GenVideo_frame/'
    save_train_dir = '/data2/ljc/dataset/GenVideo_frame/split/train/'
    save_val_dir = '/data2/ljc/dataset/GenVideo_frame/split/val/'

    index_file_names = [
        'Youku_1M_10s.txt', 'k400.txt',
        'DynamicCrafter.txt', 'I2VGEN_XL.txt', 'Latte.txt', 'OpenSora.txt', 'SEINE.txt',
        'pika.txt', 'SD.txt', 'VideoCrafter.txt', 'DynamicCrafter.txt', 'SVD.txt', 'ZeroScope.txt',
    ]

    for name in index_file_names:
        index_file_path = os.path.join(index_file_dir, name)
        save_train_path = os.path.join(save_train_dir, name)
        save_val_path = os.path.join(save_val_dir, name)
        main(index_file_path, save_train_path, save_val_path)
