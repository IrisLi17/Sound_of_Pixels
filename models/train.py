from models.models import modifyresnet18, UNet7, synthesizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


def mix_spect_input(spect_input):
    temp = np.average(spect_input, axis=0)
    return temp[np.newaxis, :]


def train1step(video_net, audio_net, syn_net, image_input, spect_input, N=2):
    """
    :param video_net: modified resnet18,
    :param audio_net: modified unet7,
    :param syn_net: a linear layer,
    :param image_input: numpy array of size (N, number_of_frames, number_of_channels, height, width), which is (N, 3, 3, 224, 224) in this project
    :param spect_input: numpy array of size (N, 1, 256, 256)
    :param N: how many videos are mixed, default to 2
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # create networks
    # video_net = modifyresnet18()
    # audio_net = UNet7()
    # syn_net = synthesizer()

    # load input
    # image_input = np.random.rand(N, 3, 3, 224, 224)
    # spect_input = np.random.rand(N, 1, 256, 256)

    # S_mix = \sum_i {S_i} / N
    synspect_input = mix_spect_input(spect_input)  # 1, 1, 256, 256
    # useful definitions
    dominant_idx = np.argmax(spect_input, axis=0)
    loss = torch.zeros(N, dtype=torch.float64)
    video_optimizer = optim.SGD(video_net.parameters(), lr=0.0001, momentum=0.9)
    audio_optimizer = optim.SGD(audio_net.parameters(), lr=0.001, momentum=0.9)
    syn_optimizer = optim.SGD(syn_net.parameters(), lr=0.001, momentum=0.9)

    image_input = torch.from_numpy(image_input).float()
    synspect_input = torch.from_numpy(synspect_input).float()
    # forward audio
    out_audio_net = audio_net.forward(synspect_input)  # size 1,K,256,256

    for i in range(N):
        # forward video
        out_video_net = video_net.forward(image_input[i, :, :, :, :])  # size (1, K, 1, 1)
        # forward synthesizer
        temp = out_video_net * out_audio_net
        temp = torch.transpose(temp, 1, 2)
        temp = torch.transpose(temp, 2, 3)
        syn_act = syn_net.forward(temp)  # sigmoid logits
        syn_act_flat = syn_act.view(-1)
        # label
        mask_truth = (dominant_idx == i).astype('float64')
        label_flat = torch.from_numpy(mask_truth.ravel()).float()
        # cross entropy loss
        loss[i] = torch.sum(-label_flat * torch.log(syn_act_flat))

    # back prop
    total_loss = torch.sum(loss)
    total_loss.backward()
    video_optimizer.step()
    audio_optimizer.step()
    syn_optimizer.step()

def test_train1step():
    N = 2
    video_net = modifyresnet18()
    audio_net = UNet7()
    syn_net = synthesizer()
    image_input = np.random.rand(N, 3, 3, 224, 224)
    spect_input = np.random.rand(N, 1, 256, 256)
    train1step(video_net, audio_net, syn_net, image_input, spect_input)

SPEC_DIR = '/data/liyunfei/dataset/audio_spectrums'
IMAGE_DIR = '/data/liyunfei/dataset/video_3frames'
INSTRUMENTS = os.listdir(SPEC_DIR)
def train_all():
    video_net = modifyresnet18()
    audio_net = UNet7()
    syn_net = synthesizer()
    for n1 in range(0,len(INSTRUMENTS)):
        instrument1 = INSTRUMENTS[n1]
        instrument_path1 = os.path.join(SPEC_DIR,instrument1)
        solo_num1 = len(os.listdir(instrument_path1))
        for n2 in range(n1,len(INSTRUMENTS)):
            instrument2 = INSTRUMENTS[n2]
            instrument_path2 = os.path.join(SPEC_DIR,instrument2)
            solo_num2 = len(os.listdir(instrument_path2))
            for s1 in range(1,solo_num1+1):
                for s2 in range(1,solo_num2+1):
                    solo_path1 = os.path.join(instrument_path1,str(s1))
                    part_num1 = len(os.listdir(solo_path1))
                    solo_path2 = os.path.join(instrument_path2,str(s2))
                    part_num2 = len(os.listdir(solo_path2))
                    for p1 in range(1,part_num1+1):
                        for p2 in range(1,part_num2+1):
                            spec1_path = os.path.join(solo_path1,str(p1)+'.npy')
                            spec2_path = os.path.join(solo_path2,str(p2)+'.npy')
                            video1_path = os.path.join(IMAGE_DIR,instrument1,str(s1),str(p1)+'.npy')
                            video2_path = os.path.join(IMAGE_DIR,instrument1,str(s2),str(p2)+'.npy')
                            spec1 = np.absolute(np.load(spec1_path))
                            spec2 = np.absolute(np.load(spec2_path))
                            spec1 = spec1[np.newaxis,:]
                            spec2 = spec2[np.newaxis,:]
                            spect_input = np.stack([spec1,spec2],axis = 0)
                            print(spect_input.shape)
                            video1 = np.load(video1_path)
                            video1 = np.transpose(video1,(0,3,1,2))
                            video2 = np.load(video2_path)
                            video2 = np.transpose(video2,(0,3,1,2))
                            image_input = np.stack([video1,video2],axis = 0)
                            print(image_input.shape)
                            #exit()
                            train1step(video_net, audio_net, syn_net, image_input, spect_input)
if __name__ == '__main__':
    test_train1step()
