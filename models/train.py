from models.models import modifyresnet18, UNet7, synthesizer
from util.datahelper import sample_input, image_normalization, load_all_training_data, sample_from_dict
from util.metrics import compute_validation
from util.waveoperate import mask2wave
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt


def mix_spect_input(spect_input):
    # temp = np.sum(spect_input, axis=0)
    # return temp[np.newaxis, :]
    return np.sum(spect_input, axis=0, keepdims=True)


def train1step(video_net, audio_net, syn_net, video_optimizer, audio_optimizer, syn_optimizer, image_input, spect_input,
               device, N=2, validate=False):
    """
    :param video_net: modified resnet18,
    :param audio_net: modified unet7,
    :param syn_net: a linear layer,
    :param image_input: numpy array of size (N, number_of_frames, number_of_channels, height, width), which is (N, 3, 3, 224, 224) in this project
    :param spect_input: numpy array of size (N, 1, 256, 256)
    :param N: how many videos are mixed, default to 2
    :return total_loss: computed cross entropy loss
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
    # loss = torch.zeros(N, dtype=torch.float64)
    total_loss = None
    estimated_spects = torch.zeros((N, 1, 256, 256))
    # video_optimizer = optim.SGD(video_net.parameters(), lr=0.0001, momentum=0.9)
    # audio_optimizer = optim.SGD(audio_net.parameters(), lr=0.001, momentum=0.9)
    # syn_optimizer = optim.SGD(syn_net.parameters(), lr=0.001, momentum=0.9)

    image_input = image_normalization(image_input).to(device)
    synspect_input = torch.from_numpy(synspect_input).to(device)
    # forward audio
    out_audio_net = audio_net.forward(synspect_input)  # size 1,K,256,256
    # print('out audio feature' + str(out_audio_net))

    for i in range(N):
        # forward video
        out_video_net = video_net.forward(image_input[i, :, :, :, :])  # size (1, K, 1, 1)
        # print('video feature ' + str(i) + str(out_video_net[0,:,0,0]))
        # forward synthesizer
        temp = out_video_net * out_audio_net
        temp = torch.transpose(temp, 1, 2)
        temp = torch.transpose(temp, 2, 3)
        syn_act = syn_net.forward(temp)  # sigmoid logits
        syn_act = torch.transpose(syn_act, 2, 3)
        syn_act = torch.transpose(syn_act, 1, 2)  # 1, 1, 256, 256
        syn_act_flat = syn_act.view(-1)
        # print('synethesizer out '+str(i)+str(syn_act_flat))
        # label
        mask_truth = (dominant_idx == i).astype('float32')
        label_flat = torch.from_numpy(mask_truth.ravel()).to(device)
        # cross entropy loss
        if total_loss is None:
            total_loss = nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
        else:
            total_loss += nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
        # loss[i] = nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
        # loss[i] = torch.sum(-label_flat * torch.log(syn_act_flat)-(1-label_flat) * torch.log(1- syn_act_flat))
        # print('synspect input size'+str(synspect_input.shape))
        # print('syn act size'+str(syn_act.shape))
        estimated_spects[i, :, :, :] = (synspect_input * syn_act)[0, :, :, :]
    # print(image_input[0,0,0,:,:])
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(image_input[0,0,0,:,:])
    # plt.subplot(1,2,2)
    # plt.imshow(image_input[1,0,0,:,:])
    video_net.zero_grad()
    audio_net.zero_grad()
    syn_net.zero_grad()
    # back prop
    # print(total_loss.grad_fn)
    # print(total_loss.grad_fn.next_functions[0][0])
    # print(total_loss.grad_fn.next_functions[0][0].next_functions[0][0])
    total_loss.backward()
    # print(list(audio_net.one_by_one.named_parameters())[0][1].grad)
    # print(2, list(audio_net.ex_double_conv2.named_parameters())[0][1].grad)
    # print(3, list(audio_net.ex_double_conv3.named_parameters())[0][1].grad)
    # print(4, list(audio_net.ex_double_conv4.named_parameters())[0][1].grad)
    # print(5, list(audio_net.ex_double_conv5.named_parameters())[0][1].grad)
    # print(6, list(audio_net.ex_double_conv6.named_parameters())[0][1].grad)
    # print(7, list(audio_net.double_conv7.named_parameters())[0][1].grad)
    # print(6, list(audio_net.double_conv6.named_parameters())[0][1].grad)
    # print(5, list(audio_net.double_conv5.named_parameters())[0][1].grad)
    # print(4, list(audio_net.double_conv4.named_parameters())[0][1].grad)
    # print('v', list(video_net.myconv2.named_parameters())[0][1].grad)
    # print('v2', list(video_net.layer2.named_parameters())[0][1].grad)
    # print('v1', list(video_net.layer1.named_parameters())[0][1].grad)
    # print(list(syn_net.linear.named_parameters())[0][1].grad)
    video_optimizer.step()
    audio_optimizer.step()
    syn_optimizer.step()
    # print(list(audio_net.named_parameters())[0][1].grad)

    if validate:
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(estimated_spects.detach().numpy()[0,0,:,:])
        # plt.subplot(2,2,2)
        # plt.imshow((dominant_idx == 0).astype('float64')[0,:,:])
        # plt.subplot(2,2,3)
        # plt.imshow(estimated_spects.detach().numpy()[1,0,:,:])
        # plt.subplot(2,2,4)
        # plt.imshow((dominant_idx == 1).astype('float64')[0,:,:])
        # plt.figure()
        # plt.imshow(synspect_input.detach().numpy()[0,0,:,:])
        # plt.show()
        return [total_loss.detach().cpu().numpy(), estimated_spects.detach().cpu().numpy()]
    else:
        return total_loss.detach().cpu().numpy()


def eval1step(video_net, audio_net, syn_net, image_input, spect_input):
    """
    :param video_net:
    :param audio_net:
    :param syn_net:
    :param image_input: only one branch of images (maybe solo is prefered?) size: (1,3,3,224,224)
    :param spect_input: corresponding spectrogram (in solo case, that is the spectogram of mixed audio) size:(1,1,256,256)
    :return:
    """
    image_input = torch.from_numpy(image_input).float()
    synspect_input = torch.from_numpy(spect_input).float()
    # forward audio
    out_audio_net = audio_net.forward(synspect_input)  # size 1,K,256,256
    # forward video, get pixel level features
    out_video_net = video_net.forward(image_input, mode='eval')


def test_train1step():
    N = 2
    video_net = modifyresnet18()
    audio_net = UNet7()
    syn_net = synthesizer()
    spec_dir = '/data/liyunfei/dataset/audio_spectrums'
    image_dir = '/data/liyunfei/dataset/video_3frames'
    [spect_input, image_input] = sample_input(spec_dir, image_dir, 'train')
    # image_input = np.random.rand(N, 3, 3, 224, 224)
    # spect_input = np.random.rand(N, 1, 256, 256)
    return train1step(video_net, audio_net, syn_net, image_input, spect_input, validate=True)


SPEC_DIR = '/data/liyunfei/dataset/audio_spectrums'
IMAGE_DIR = '/data/liyunfei/dataset/video_3frames'


def train_all(spec_dir, image_dir, num_epoch=10, validate_freq=10000, log_freq=100, log_dir=None, model_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_net = modifyresnet18().to(device)
    audio_net = UNet7().to(device)
    syn_net = synthesizer().to(device)
    # video_optimizer = optim.SGD(video_net.parameters(), lr=0.0001, momentum=0.9)
    myconv_params = list(map(id, video_net.myconv2.parameters()))
    base_params = filter(lambda p: id(p) not in myconv_params,
                         video_net.parameters())
    video_optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': video_net.myconv2.parameters(), 'lr': 0.001},
    ], lr=0.0001, momentum=0.9)
    audio_optimizer = optim.SGD(audio_net.parameters(), lr=0.001, momentum=0.9)
    syn_optimizer = optim.SGD(syn_net.parameters(), lr=0.001, momentum=0.9)
    [spec_data, image_data] = load_all_training_data(spec_dir, image_dir)
    print('Data loaded!')
    # video_net.parameters()
    phases = ['train', 'validate', 'test']
    for epoch in range(num_epoch):
        total_loss = 0.0
        count = 0
        for t in itertools.count():
            if t > 50000:
                break
            if t % validate_freq == 0:
                # [spect_input, image_input] = sample_input(spec_dir, image_dir, 'validate')
                [spect_input, image_input] = sample_from_dict(spec_data, image_data)
                if not (spect_input is None or image_input is None):
                    [loss, estimated_spects] = train1step(video_net, audio_net, syn_net, video_optimizer,
                                                          audio_optimizer, syn_optimizer, image_input, spect_input,
                                                          device, validate=True)
                    total_loss += loss
                    count += 1
                    # convert spects to wav
                    wav_input = np.stack([mask2wave(spect_input[i, 0, :, :]) for i in range(spect_input.shape[0])],
                                         axis=0)  # N, nsample
                    wav_mixed = np.reshape(mask2wave(mix_spect_input(spect_input)), (1, -1))  # 1, nsample
                    wav_estimated = np.stack(
                        [mask2wave(estimated_spects[i, 0, :, :]) for i in range(estimated_spects.shape[0])],
                        axis=0)  # N, nsample
                    # print('wav input shape: ' + str(wav_input.shape))
                    # print('wav mixed shape: ' + str(wav_mixed.shape))
                    # print('wav estimated shape: ' + str(wav_estimated.shape))
                    [nsdr, sir, sar] = compute_validation(wav_input, wav_estimated, wav_mixed)
            else:
                # [spect_input, image_input] = sample_input(spec_dir, image_dir, 'train')
                [spect_input, image_input] = sample_from_dict(spec_data, image_data)
                if not (spect_input is None or image_input is None):
                    total_loss += train1step(video_net, audio_net, syn_net, video_optimizer, audio_optimizer,
                                             syn_optimizer, image_input, spect_input, device,
                                             validate=False)
                    count += 1
            if t % log_freq == 0:
                print("epoch %d" % epoch)
                print("steps %d" % t)
                print("average loss %f" % (total_loss / count))
                if t % validate_freq == 0:
                    if not (nsdr is None or sir is None or sar is None):
                        print("nsdr %f, %f" % (nsdr[0], nsdr[1]))
                        print("sir %f, %f" % (sir[0], sir[1]))
                        print("sar %f, %f" % (sar[0], sar[1]))
                sys.stdout.flush()
                if not os.path.exists(os.path.join(log_dir, 'log.csv')):
                    os.mkdir(log_dir)
                    with open(os.path.join(log_dir, 'log.csv'), 'a', newline='') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=',',
                                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
                        title = ['epoch', 'step', 'ave_loss', 'nsdr0', 'nsdr1', 'sir0', 'sir1', 'sar0', 'sar1']
                        spamwriter.writerow(title)
                with open(os.path.join(log_dir, 'log.csv'), 'a', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
                    if t % validate_freq == 0 and not (nsdr is None or sir is None or sar is None):
                        data = [epoch, t, total_loss / count, nsdr[0], nsdr[1], sir[0], sir[1], sar[0], sar[1]]
                    else:
                        data = [epoch, t, total_loss / count, None, None, None, None, None, None]
                    spamwriter.writerow(data)

                total_loss = 0.0
                count = 0

                # for n1 in range(0, len(INSTRUMENTS)):
                #     instrument1 = INSTRUMENTS[n1]
                #     instrument_path1 = os.path.join(SPEC_DIR, instrument1)
                #     solo_num1 = len(os.listdir(instrument_path1))
                #     for n2 in range(n1, len(INSTRUMENTS)):
                #         instrument2 = INSTRUMENTS[n2]
                #         instrument_path2 = os.path.join(SPEC_DIR, instrument2)
                #         solo_num2 = len(os.listdir(instrument_path2))
                #         for s1 in range(1, solo_num1 + 1):
                #             for s2 in range(1, solo_num2 + 1):
                #                 solo_path1 = os.path.join(instrument_path1, str(s1))
                #                 part_num1 = len(os.listdir(solo_path1))
                #                 solo_path2 = os.path.join(instrument_path2, str(s2))
                #                 part_num2 = len(os.listdir(solo_path2))
                #                 for p1 in range(1, part_num1 + 1):
                #                     for p2 in range(1, part_num2 + 1):
                #                         spec1_path = os.path.join(solo_path1, str(p1) + '.npy')
                #                         spec2_path = os.path.join(solo_path2, str(p2) + '.npy')
                #                         video1_path = os.path.join(IMAGE_DIR, instrument1, str(s1), str(p1) + '.npy')
                #                         video2_path = os.path.join(IMAGE_DIR, instrument1, str(s2), str(p2) + '.npy')
                #                         spec1 = np.absolute(np.load(spec1_path))
                #                         spec2 = np.absolute(np.load(spec2_path))
                #                         spec1 = spec1[np.newaxis, :]
                #                         spec2 = spec2[np.newaxis, :]
                #                         spect_input = np.stack([spec1, spec2], axis=0)
                #                         # print(spect_input.shape)
                #                         video1 = np.load(video1_path)
                #                         video1 = np.transpose(video1, (0, 3, 1, 2))
                #                         video2 = np.load(video2_path)
                #                         video2 = np.transpose(video2, (0, 3, 1, 2))
                #                         image_input = np.stack([video1, video2], axis=0)
                #                         # print(image_input.shape)
                #                         # exit()
                #                         total_loss = train1step(video_net, audio_net, syn_net, image_input, spect_input)
                #                         # save params
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(video_net.state_dict(), os.path.join(model_dir, 'video_net_params.pkl'))
        torch.save(audio_net.state_dict(), os.path.join(model_dir, 'audio_net_params.pkl'))
        torch.save(syn_net.state_dict(), os.path.join(model_dir, 'syn_net_params.pkl'))
        print("model saved to " + str(model_dir) + '\n')


if __name__ == '__main__':
    test_train1step()
