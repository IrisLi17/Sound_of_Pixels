import os
import matplotlib.pyplot as plt
from util.datahelper import load_data_from_video
from models.train import train_all, test_all, test_test1step, test_evaluate
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
    parser.add_argument('--log_dir', default='../testlog')
    parser.add_argument('--testresult_dir', default='../test_result')
    parser.add_argument('--model_dir', default='../model')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--validate', type=int, default=0)
    args = parser.parse_args()
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    test_audio_dir = r'D:\huyb\std\testset25\gt_audio'
    test_video_dir = r'D:\huyb\std\testset25\testimage'
    test_result_dir = args.testresult_dir
    ground_truth_dir = '../testset25_audio_gt/evaluate'
    test_all(test_video_dir,test_audio_dir,test_result_dir,batch_size=1,
            log_dir=log_dir,model_dir=args.model_dir,test_type='test25')
    # test_evaluate(ground_truth_dir,test_result_dir)


if __name__ == '__main__':
    main()
