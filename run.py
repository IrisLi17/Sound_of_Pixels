import os
import matplotlib.pyplot as plt
from util.datahelper import load_data_from_video
from models.train import train_all, test_all
import datetime
import argparse


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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spect_dir', help='spectrogram directory', default='../audio_spectrums_linear')
    parser.add_argument('--image_dir', help='image directory', default='../video_3frames')
    parser.add_argument('--log_dir', default='../log')
    parser.add_argument('--testresult_dir',default = '../test_result')
    parser.add_argument('--model_dir', default='../model')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--validate', type=int, default=0)
    args = parser.parse_args()
    SPEC_DIR = args.spect_dir
    IMAGE_DIR = args.image_dir
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # train_all(SPEC_DIR, IMAGE_DIR, batch_size=args.batch_size, log_dir=log_dir, model_dir=args.model_dir,
    #          validate=bool(args.validate))
    test_audio_dir = '../dataset1/dataset/audios/duet'
    test_video_dir = '../dataset1/dataset/videos/duet'
    test_result_dir = args.testresult_dir
    test_all(test_video_dir,test_audio_dir,test_result_dir,batch_size=1,
            log_dir=log_dir,model_dir=args.model_dir)


if __name__ == '__main__':
    main()
