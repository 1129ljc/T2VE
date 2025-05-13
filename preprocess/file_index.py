import os
import argparse


def main(root_dir, txt_file):
    txt_file_open = open(txt_file, mode='w', encoding='utf-8')
    names = sorted(os.listdir(root_dir))
    for name in names:
        root_name_dir = os.path.join(root_dir, name)
        pic_num = len(os.listdir(root_name_dir))
        str_s = root_name_dir + ' ' + str(pic_num) + '\n'
        txt_file_open.write(str_s)
    txt_file_open.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='video frames save dir', required=True)
    parser.add_argument('--txt_file', type=str, help='index file save path', required=True)
    args = parser.parse_args()
    main(args.root_dir, args.txt_file)
