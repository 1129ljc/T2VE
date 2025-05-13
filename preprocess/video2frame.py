import os
import math
import argparse
from moviepy.editor import VideoFileClip


def process_video(video_path, image_path):
    video_name = os.path.basename(video_path)
    if os.path.exists(image_path):
        print(image_path, 'exists!')
    else:
        try:
            try:
                video = VideoFileClip(video_path)
                video_length = video.duration
                print(video_name, f"视频长度为：{video_length} 秒")
                os.makedirs(image_path)

                if video_length >= 4:
                    inter_val = 2
                    os.system(f"ffmpeg -loglevel quiet -i {video_path} -r {inter_val} {image_path}/%d.jpg")
                else:
                    inter_val = math.ceil(8 / video_length)
                    os.system(f"ffmpeg -loglevel quiet -i {video_path} -r {inter_val} {image_path}/%d.jpg")

            except Exception as e:
                print("发生异常：", str(e))
        except:
            print("Skip")


def main(video_dir, save_dir):
    video_names = sorted(os.listdir(video_dir))
    for index in range(len(video_names)):
        video_name = video_names[index]
        video_path = os.path.join(video_dir, video_name)
        image_path = os.path.join(save_dir, video_name.split('.')[0])
        process_video(video_path, image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='video save dir', required=True)
    parser.add_argument('--save_dir', type=str, help='video frames save dir', required=True)
    args = parser.parse_args()
    main(args.video_dir, args.save_dir)
