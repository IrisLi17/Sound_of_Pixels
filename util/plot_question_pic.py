import torch
import os 
import sys
import math
import cv2
import numpy as np
sys.path.append(r'D:\huyb\std\Sound_of_Pixels')
from models.models import modifyresnet18, UNet7, synthesizer
from util.datahelper import image_normalization
from util.split_image_hough import split_image
from librosa import amplitude_to_db
import matplotlib.pyplot as plt


def mix_spect_input(spect_input):
    # temp = np.sum(spect_input, axis=0)
    # return temp[np.newaxis, :]
    return amplitude_to_db(np.absolute(np.sum(spect_input, axis=0)), ref=np.max)

BLOCK_TIME = 66302/11000
FPS=24
video_block_length = math.floor(BLOCK_TIME*FPS)
FRAME_INDEX = [0, 70, 140]
frame_each_block = math.floor(video_block_length/10)

names = ['1', '2']
features = np.zeros((len(names),16))
batch_size = 1
model_dir = r'D:\huyb\std\model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_net = modifyresnet18(batch_size).to(device)
audio_net = UNet7().to(device)
syn_net = synthesizer().to(device)
if os.path.exists(os.path.join(model_dir, '18video_net_params.pkl')) and os.path.exists(
        os.path.join(model_dir, '18audio_net_params.pkl')) and os.path.exists(
    os.path.join(model_dir, '18syn_net_params.pkl')):
    print('load params!')
    video_net.load_state_dict(torch.load(os.path.join(model_dir, '18video_net_params.pkl')))
    audio_net.load_state_dict(torch.load(os.path.join(model_dir, '18audio_net_params.pkl')))
    syn_net.load_state_dict(torch.load(os.path.join(model_dir, '18syn_net_params.pkl')))
    video_net.eval()
    audio_net.eval()
    syn_net.eval()

image_dir = r'D:\huyb\std\testset25\testimage\saxophone_2_violin_2'

frameCount = int(os.listdir(image_dir)[-1][0:-4])
block_num = math.floor(frameCount/BLOCK_TIME/FPS*10)
print("block_num:",block_num)
for i in range(block_num):
    index_str = [str(math.floor(x/10)+1+i*frame_each_block).zfill(6) + '.jpg' for x in FRAME_INDEX] 
    final = np.empty((2, len(FRAME_INDEX), 224, 224, 3),np.dtype('uint8'))
    for idx in range(len(index_str)):
        frame_dir = os.path.join(image_dir,index_str[idx])
        frame = np.array(cv2.imread(frame_dir),dtype='uint8')
        gap = split_image(frame_dir)[0]
                # print("gap",gap)
        final[0, idx, :, :, :] = cv2.resize(frame[ :, 0:gap, :], (224, 224)) 
        final[1, idx, :, :, :] = cv2.resize(frame[:, gap:-1, :], (224, 224))
        if i==0 and idx==0:
            plt.subplot(1,2,1)
            plt.imshow(final[0, idx, :, :, :])
            plt.subplot(1,2,2)
            plt.imshow(final[1, idx, :, :, :])
            plt.show()
                # exit()
                                        
        video_input = np.transpose(final,(0,1,4,2,3))
        video_input = image_normalization(video_input).to(device)
        for i in range(2):
            out_video_net = video_net.forward(video_input[i, :, :, :, :])
            if i==0 or (i==1 and idx==len(names)-2):
                features[i,:] += out_video_net[0,:,0,0].detach().cpu().numpy()

## add a tail for image data
final = np.empty((2, len(FRAME_INDEX), 224, 224, 3),np.dtype('uint8'))
frame_dir = os.path.join(image_dir,os.listdir(image_dir)[-1])
for idx in range(3):
    frame = np.array(cv2.imread(frame_dir),dtype='uint8')
    gap = split_image(frame_dir)[0]
    final[0, idx, :, :, :] = cv2.resize(frame[ :, 0:gap, :], (224, 224)) 
    final[1, idx, :, :, :] = cv2.resize(frame[:, gap:-1, :], (224, 224))
video_input = np.transpose(final,(0,1,4,2,3))
video_input = image_normalization(video_input).to(device)
for i in range(2):
    out_video_net = video_net.forward(video_input[i, :, :, :, :])
    if i==0 or (i==1 and idx==len(names)-2):
        features[i,:] += out_video_net[0,:,0,0].detach().cpu().numpy()

features[:,:] /= block_num
print("finish compute")
for i in range(len(names)):
    plt.figure()
    plt.bar(range(16),features[i,:])
    plt.title('feature of '+names[i])
    # plt.savefig(names[i]+'.jpg')
    plt.show()
print("save bars")
