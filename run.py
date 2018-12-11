import os
import matplotlib.pyplot as plt
from util.datahelper import load_data_from_video
from models.train import train_all
import datetime


def main():
    # video_path = '../demo_project/testvideo'
    # audio_path = '../demo_project/gt_audio'
    # Videoname = os.path.join(video_path, "1.mp4")
    # audioname = os.path.join(audio_path, "1.wav")
    # video, audio = load_data_from_video(Videoname, audioname, 10)
    # print("video shape: " + str(video.shape)) # (183, 224, 224, 3)
    # print("audio shape: " + str(audio.shape)) # (1, 2703360)
    # plt.figure(1)
    # plt.subplot(2,1,1)
    # plt.imshow(video[100,:,:,:])
    # plt.subplot(2,1,2)
    # plt.imshow(video[101,:,:,:])
    # plt.show()
    SPEC_DIR = '/data/liyunfei/dataset/audio_spectrums'
    IMAGE_DIR = '/data/liyunfei/dataset/video_3frames'
    log_dir = os.path.join('/data/liyunfei/dataset/log', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    train_all(SPEC_DIR, IMAGE_DIR, log_dir=log_dir, model_dir='/data/liyunfei/dataset/model')


if __name__ == '__main__':
    main()
