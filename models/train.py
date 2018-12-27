from models.models import modifyresnet18, UNet7, synthesizer
from util.datahelper import sample_input, image_normalization, load_all_training_data, sample_from_dict , load_test_data
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
import librosa
import math
import cv2
import mir_eval
from librosa import amplitude_to_db
from librosa.output import write_wav
import librosa.display
from sklearn.cluster import KMeans


def mix_spect_input(spect_input):
    # temp = np.sum(spect_input, axis=0)
    # return temp[np.newaxis, :]
    return amplitude_to_db(np.absolute(np.sum(spect_input, axis=0)), ref=np.max)


def train1step(video_net, audio_net, syn_net, video_optimizer, audio_optimizer, syn_optimizer, image_input, spect_input,
               device, N=2, validate=False):
    """
    :param video_net: modified resnet18,
    :param audio_net: modified unet7,
    :param syn_net: a linear layer,
    :param image_input: numpy array of size (N, number_of_frames * batch_size, number_of_channels, height, width), which is (N, 3, 3, 224, 224) in this project
    :param spect_input: numpy array of size (N, batch_size, 1, 256, 256)
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
    synspect_input = mix_spect_input(spect_input)  # batch_size, 1, 256, 256
    # useful definitions
    dominant_idx = np.argmax(np.absolute(spect_input), axis=0)  # batch_size, 1, 256, 256
    # loss = torch.zeros(N, dtype=torch.float64)
    total_loss = None
    estimated_spects = torch.zeros((N, video_net.batch_size, 1, 256, 256))
    # video_optimizer = optim.SGD(video_net.parameters(), lr=0.0001, momentum=0.9)
    # audio_optimizer = optim.SGD(audio_net.parameters(), lr=0.001, momentum=0.9)
    # syn_optimizer = optim.SGD(syn_net.parameters(), lr=0.001, momentum=0.9)

    image_input = image_normalization(image_input).to(device)
    synspect_input = torch.from_numpy(synspect_input).to(device)
    # forward audio
    out_audio_net = audio_net.forward(synspect_input)  # size batch_size,K,256,256
    # print('out audio feature' + str(out_audio_net))

    for i in range(N):
        # forward video
        out_video_net = video_net.forward(image_input[i, :, :, :, :])  # size (batch_size, K, 1, 1)
        # print('video feature ' + str(i) + str(out_video_net[0,:,0,0]))
        # forward synthesizer
        temp = out_video_net * out_audio_net
        temp = torch.transpose(temp, 1, 2)
        temp = torch.transpose(temp, 2, 3)
        syn_act = syn_net.forward(temp)  # sigmoid logits
        syn_act = torch.transpose(syn_act, 2, 3)
        syn_act = torch.transpose(syn_act, 1, 2)  # batch_size, 1, 256, 256

        print_syn = syn_act[0,0,:,:]

        syn_act_flat = syn_act.view(-1)
        # print('synethesizer out '+str(i)+str(syn_act_flat))
        # label
        mask_truth = (dominant_idx == i).astype('float32')
        label_flat = torch.from_numpy(mask_truth.ravel()).to(device)
        # cross entropy loss
        if total_loss is None:
            total_loss = nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
            '''
            temp =  nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
            print("loss:",temp)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(print_syn.detach().cpu().numpy())
            plt.subplot(1,2,2)
            plt.imshow(mask_truth[0,0,:,:])
            plt.show()
            '''
            
        else:
            '''
            temp =  nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
            print("loss:",temp)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(print_syn.detach().cpu().numpy())
            plt.subplot(1,2,2)
            plt.imshow(mask_truth[0,0,:,:])
            plt.show()
            '''
            
            total_loss += nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
        # loss[i] = nn.functional.binary_cross_entropy(syn_act_flat, label_flat)
        # loss[i] = torch.sum(-label_flat * torch.log(syn_act_flat)-(1-label_flat) * torch.log(1- syn_act_flat))
        # print('synspect input size'+str(synspect_input.shape))
        # print('syn act size'+str(syn_act.shape))
        # estimated_spects[i, :, :, :, :] = synspect_input * syn_act
        estimated_spects[i, :, :, :, :] = syn_act
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
    if not validate:
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
        return total_loss.detach().cpu().numpy()
    else:
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(estimated_spects.detach().numpy()[0,0,0,:,:])
        # plt.subplot(2,2,2)
        # plt.imshow((dominant_idx == 0).astype('float64')[0,:,:])
        # plt.subplot(2,2,3)
        # plt.imshow(estimated_spects.detach().numpy()[1,0,:,:])
        # plt.subplot(2,2,4)
        # plt.imshow((dominant_idx == 1).astype('float64')[0,:,:])
        # plt.figure()
        # plt.imshow(synspect_input.detach().numpy()[0,0,:,:])
        # plt.show()
        ground_truth = np.asarray([(dominant_idx == i)[0, 0, :, :] for i in range(N)])
        return [total_loss.detach().cpu().numpy(), estimated_spects.detach().cpu().numpy(), ground_truth]


def test1step(video_net, audio_net, syn_net, image_input, spect_input, device):
    """
    :param video_net:
    :param audio_net:
    :param syn_net:
    :param image_input: only one branch of images (duet); size: (1, number_of_frames * batch_size, number_of_channels, height, width)
    :param spect_input: corresponding duet spectrogram; size: (1, batch_size, 1, 256, 256)
    :return:
    """
    # image_input = image_input[0, :, :, :, :]
    image_input = image_normalization(image_input).to(device)
    synspect_input = spect_input[0, :, :, :, :]
    synspect_input = amplitude_to_db(np.absolute(synspect_input),ref = np.max)
    synspect_input = torch.from_numpy(synspect_input).to(device)
    # forward audio
    # print("audio_input",synspect_input)
    out_audio_net = audio_net.forward(synspect_input)  # size batch_size,K,256,256
    # forward video, get pixel level features
    out_video_net = video_net.forward(image_input[0,:,:,:,:], mode='tes')  # size batch_size, K, 14, 14
    estimated_spects = torch.zeros((video_net.batch_size, out_video_net.shape[-2], out_video_net.shape[-1], 1,
                                    spect_input.shape[-2], spect_input.shape[-1]))
    for idx1 in range(out_video_net.shape[-2]):
        for idx2 in range(out_video_net.shape[-1]):
            temp = out_video_net[:, :, idx1, idx2]
            temp = temp[:,:,np.newaxis,np.newaxis]
            # print(temp.shape,out_audio_net.shape)
            temp = temp * out_audio_net            
            temp = torch.transpose(temp, 1, 2)
            temp = torch.transpose(temp, 2, 3)
            syn_act = syn_net.forward(temp)  # sigmoid logits
            syn_act = torch.transpose(syn_act, 2, 3)
            syn_act = torch.transpose(syn_act, 1, 2)  # batch_size, 1, 256, 256
            estimated_spects[:, idx1, idx2, :, :, :] = syn_act
    return estimated_spects.detach().cpu().numpy()


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


def test_test1step(batch_size=1, model_dir='../model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_net = modifyresnet18(batch_size).to(device)
    audio_net = UNet7().to(device)
    syn_net = synthesizer().to(device)
    video_net.load_state_dict(torch.load(os.path.join(model_dir, 'video_net_params.pkl')))
    audio_net.load_state_dict(torch.load(os.path.join(model_dir, 'audio_net_params.pkl')))
    syn_net.load_state_dict(torch.load(os.path.join(model_dir, 'syn_net_params.pkl')))
    video_net.eval()
    audio_net.eval()
    syn_net.eval()
    video_input = np.load('/home/thu-skyworks/irisli/Projects/video_input_left.npy')
    audio_input = np.load('/home/thu-skyworks/irisli/Projects/audio_input.npy')
    estimated_spects = test1step(video_net, audio_net, syn_net, video_input,
                                 amplitude_to_db(np.absolute(audio_input), ref=np.max),
                                 device)
    np.save('estimated_spects_left', estimated_spects)


SPEC_DIR = '/data/liyunfei/dataset/audio_spectrums'
IMAGE_DIR = '/data/liyunfei/dataset/video_3frames'


def train_all(spec_dir, image_dir, num_epoch=10, steps_per_epoch = 50000, batch_size=1, N=2, validate_freq=10000, log_freq=100, log_dir=None,
              model_dir=None, validate=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_net = modifyresnet18(batch_size).to(device)
    audio_net = UNet7().to(device)
    syn_net = synthesizer().to(device)
    if os.path.exists(os.path.join(model_dir, '4video_net_params.pkl')) and os.path.exists(
            os.path.join(model_dir, '4audio_net_params.pkl')) and os.path.exists(
        os.path.join(model_dir, '4syn_net_params.pkl')):
        print('load params!')
        video_net.load_state_dict(torch.load(os.path.join(model_dir, '4video_net_params.pkl')))
        audio_net.load_state_dict(torch.load(os.path.join(model_dir, '4audio_net_params.pkl')))
        syn_net.load_state_dict(torch.load(os.path.join(model_dir, '4syn_net_params.pkl')))
        if validate:
            video_net.eval()
            audio_net.eval()
            syn_net.eval()
        else:
            video_net.train()
            audio_net.train()
            syn_net.train()

    
    if not validate:
        video_optimizer = optim.SGD(video_net.parameters(), lr=0.0001, momentum=0.9)
        myconv_params = list(map(id, video_net.myconv2.parameters()))
        base_params = filter(lambda p: id(p) not in myconv_params,
                            video_net.parameters())
        video_optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': video_net.myconv2.parameters(), 'lr': 0.001},
        ], lr=0.0001, momentum=0.9)
        # video_optimizer = optim.SGD(video_net.myconv2.parameters(), lr=0.001, momentum=0.9)
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
                if t > steps_per_epoch:
                    break
                if t % validate_freq == 0:
                    # [spect_input, image_input] = sample_input(spec_dir, image_dir, 'validate')
                    image_input = np.zeros((N, 3 * batch_size, 3, 224, 224), dtype='float32')
                    _spect_input = []
                    for bidx in range(batch_size):
                        [spect_input_comp, image_input_mini] = sample_from_dict(spec_data, image_data)
                        
                        # _spect_input.append(amplitude_to_db(np.absolute(spect_input_comp), ref=np.max))
                        _spect_input.append(spect_input_comp)
                        # dB shouln't be take here

                        # _image_input.append(image_input_mini)
                        image_input[:, 3 * bidx:3 * bidx + 3, :, :, :] = image_input_mini
                    spect_input = np.transpose(np.stack(_spect_input, axis=0),
                                               (1, 0, 2, 3, 4))  # expect shape (N, batch_size, 1, 256, 256)
                    # print('spect_input shape', spect_input.shape)
                    # print('image input shape', image_input.shape)
                    if not (spect_input is None or image_input is None):
                        [loss, estimated_spects, ground_truth] = train1step(video_net, audio_net, syn_net,
                                                                            video_optimizer,
                                                                            audio_optimizer, syn_optimizer, image_input,
                                                                            spect_input,
                                                                            device, validate=True)
                        total_loss += loss
                        count += 1
                        # convert spects to wav
                        mixed_spect = np.sum(spect_input_comp,axis=0)[0,:,:]
                        # mixed_spect = mix_spect_input(spect_input_comp)[0, :, :]
                        wav_input_ground = np.stack(
                            [mask2wave(spect_input_comp[i, 0, :, :], type='linear') for i in
                             range(spect_input_comp.shape[0])],
                            axis=0)  # N, nsample
                        wav_input_cal = np.stack([mask2wave(ground_truth[i, :, :] * mixed_spect, type='linear') for i in
                                                  range(ground_truth.shape[0])], axis=0)
                        wav_mixed = np.reshape(mask2wave(mixed_spect, type='linear'),
                                               (1, -1))  # 1, nsample
                        wav_estimated = np.stack(
                            [mask2wave(estimated_spects[i, 0, 0, :, :] * mixed_spect, type='linear') for i in
                             range(estimated_spects.shape[0])],
                            axis=0)  # N, nsample
                        # print('wav input shape: ' + str(wav_input.shape))
                        # print('wav mixed shape: ' + str(wav_mixed.shape))
                        # print('wav estimated shape: ' + str(wav_estimated.shape))
                        [nsdr, sir, sar] = compute_validation(wav_input_ground, wav_estimated, wav_mixed)
                else:
                    # [spect_input, image_input] = sample_input(spec_dir, image_dir, 'train')
                    image_input = np.zeros((N, 3 * batch_size, 3, 224, 224), dtype='float32')
                    _spect_input = []
                    for bidx in range(batch_size):
                        [spect_input_comp, image_input_mini] = sample_from_dict(spec_data, image_data)

                        _spect_input.append(spect_input_comp)
                        # dB should not be taken here either!

                        # _image_input.append(image_input_mini)
                        image_input[:, 3 * bidx:3 * bidx + 3, :, :, :] = image_input_mini
                    spect_input = np.transpose(np.stack(_spect_input, axis=0),
                                               (1, 0, 2, 3, 4))  # expect shape (N, batch_size, 1, 256, 256)
                    # print('spect_input shape', spect_input.shape)
                    # print('image input shape', image_input.shape)
                    if not (spect_input is None or image_input is None):
                        temp = total_loss
                        total_loss += train1step(video_net, audio_net, syn_net, video_optimizer, audio_optimizer,
                                                 syn_optimizer, image_input, spect_input, device,
                                                 validate=False)
                        # print("this loss:",total_loss-temp)
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
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            torch.save(video_net.state_dict(), os.path.join(model_dir, str(epoch) + 'video_net_params.pkl'))
            torch.save(audio_net.state_dict(), os.path.join(model_dir, str(epoch) + 'audio_net_params.pkl'))
            torch.save(syn_net.state_dict(), os.path.join(model_dir, str(epoch) + 'syn_net_params.pkl'))
            print("model saved to " + str(model_dir) + '\n')
    else:
        [spec_data, image_data] = load_all_training_data(spec_dir, image_dir)
        print('Data loaded!')
        while (input() != 'q'):
            image_input = np.zeros((N, 3 * batch_size, 3, 224, 224), dtype='float32')
            '''
            _spect_input = []
            for bidx in range(batch_size):
                [spect_input_comp, image_input_mini] = sample_from_dict(spec_data, image_data)
                        
                _spect_input.append(spect_input_comp)
                # dB shouln't be take here
                image_input[:, 3 * bidx:3 * bidx + 3, :, :, :] = image_input_mini
            spect_input = np.transpose(np.stack(_spect_input, axis=0),
                                               (1, 0, 2, 3, 4))
            '''
            [spect_input_comp, image_input_mini] = sample_from_dict(spec_data, image_data)  # N,1,256,256; N,3,3,224,224
            spect_input_mini = np.transpose(spect_input_comp[np.newaxis, :], (1, 0, 2, 3, 4))

            # print(spect_input_mini.shape)
            # print(image_input_mini.shape)

            # the following part is to test the validate part with certain data
            '''
            video_input1 = np.load('D:\\huyb\\std\\video_3frames\\flute\\1\\8.npy')
            video_input2 = np.load('D:\\huyb\\std\\video_3frames\\cello\\1\\7.npy')
            video_input1 = np.transpose(video_input1,(0,3,1,2))
            video_input2 = np.transpose(video_input2,(0,3,1,2))
            audio_input1 = np.load('D:\\huyb\\std\\audio_spectrums_linear\\flute\\1\\8.npy')
            audio_input2 = np.load('D:\\huyb\\std\\audio_spectrums_linear\\cello\\1\\7.npy')
            audio_input = []
            audio_input.append(audio_input1[np.newaxis,:])
            audio_input.append(audio_input2[np.newaxis,:])
            spect_input_comp = np.stack(audio_input,axis=0)
            # spect_input_mini = amplitude_to_db(np.absolute(spect_input_comp), ref=np.max)
            spect_input_mini = spect_input_comp
            spect_input_mini = np.transpose(spect_input_mini[np.newaxis,:], (1, 0, 2, 3, 4))
            video_input = []
            video_input.append(video_input1)
            video_input.append(video_input2)
            image_input_mini = np.stack(video_input,axis=0)
            print(spect_input_mini.shape)
            print(image_input_mini.shape)
            '''

            [_, estimated_masks, ground_truth] = train1step(video_net, audio_net, syn_net, None, None, None,
                                                            image_input_mini, spect_input_mini,
                                                            device, N=2, validate=True)
                                                            
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(estimated_masks[0,0,0,:,:])
            plt.subplot(2,2,2)
            plt.imshow(ground_truth[0])
            plt.title('ground truth 0')
            plt.subplot(2,2,3)
            plt.imshow(estimated_masks[1,0,0,:,:])
            plt.subplot(2,2,4)
            plt.imshow(ground_truth[1])
            plt.title('ground truth 1')
            plt.show()
            

            mixed_spect = np.sum(spect_input_comp,axis=0)
            wav_input_ground = np.stack(
                [mask2wave(spect_input_comp[i, 0, :, :], type='linear') for i in range(spect_input_comp.shape[0])],
                axis=0)  # N, nsample
            wav_input_cal = np.stack([mask2wave(ground_truth[i, :, :] * mixed_spect, type='linear') for i in
                                      range(ground_truth.shape[0])], axis=0)
            wav_mixed = np.reshape(mask2wave(mixed_spect, type='linear'),
                                   (1, -1))  # 1, nsample
            wav_estimated = np.stack(
                [mask2wave(estimated_masks[i, 0, 0, :, :] * mixed_spect, type='linear') for i in
                 range(estimated_masks.shape[0])],
                axis=0)  # N, nsample
            [nsdr, sir, sar] = compute_validation(wav_input_ground, wav_estimated, wav_mixed)
            print("nsdr %f, %f" % (nsdr[0], nsdr[1]))
            print("sir %f, %f" % (sir[0], sir[1]))
            print("sar %f, %f" % (sar[0], sar[1]))

            fs = 11000
            for i in range(2):
                write_wav('input_cal1' + str(i) + '.wav', wav_input_cal[i, :], fs)
                write_wav('input_ground' + str(i) + '.wav', wav_input_ground[i, :], fs)
                write_wav('estimated' + str(i) + '.wav', wav_estimated[i, :], fs)
            write_wav('mixed.wav', wav_mixed[0, :], fs)


def test_all(video_dir, audio_dir, result_dir, batch_size=1, log_dir=None, model_dir=None,test_type='validate'):                    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_net = modifyresnet18(batch_size).to(device)
    audio_net = UNet7().to(device)
    syn_net = synthesizer().to(device)
    clust_estimator = KMeans(n_clusters=2)
    assert os.path.exists(os.path.join(model_dir, '4video_net_params.pkl'))
    assert os.path.exists(os.path.join(model_dir, '4audio_net_params.pkl'))
    assert os.path.exists(os.path.join(model_dir, '4syn_net_params.pkl'))
    video_net.load_state_dict(torch.load(os.path.join(model_dir, '4video_net_params.pkl')))
    audio_net.load_state_dict(torch.load(os.path.join(model_dir, '4audio_net_params.pkl')))
    syn_net.load_state_dict(torch.load(os.path.join(model_dir, '4syn_net_params.pkl')))
    # TODO load testing data and call `test1step`
    video_net.eval()
    audio_net.eval()
    syn_net.eval()

    [video_dic, audio_dic] = load_test_data(video_dir, audio_dir,test_type=test_type)
    if test_type=='test25':
        if(len(video_dic.keys())==len(audio_dic.keys())):
            files = video_dic.keys()
        else:
            print("different number of instruments in video and audio!")
            print(len(video_dic.keys()))
            print(len(audio_dic.keys()))
            return
        for onefile in files:
            video_data = np.array(video_dic[onefile])
            audio_data = np.array(audio_dic[str(onefile)+'.wav'])
            print(video_data.shape)
            print(audio_data.shape)
            block_num = video_data.shape[0]
            estimated_wav = []
            for i in range(block_num):
                for k in range(2):
                    video_input = video_data[i,k,:,:,:,:][np.newaxis,:]
                    audio_input = audio_data[i,:,:,:,:,:]
                    mask = test1step(video_net,audio_net,syn_net,video_input,
                                                audio_input,
                                                device)
                    if i==0:
                        estimated_wav.append(mask2wave(mask[0,0,0,0,:,:] * audio_input,'linear'))
                    else:  
                        estimated_wav[k] = np.append(estimated_wav[k],mask2wave(mask[0,0,0,0,:,:] * audio_input,'linear'))
            estimated_wav = np.array(estimated_wav)
            print(estimated_wav.shape)
            write_wav(os.path.join(result_dir,onefile+'_1.wav'),estimated_wav[0,:],11000)
            write_wav(os.path.join(result_dir,onefile+'_2.wav'),estimated_wav[1,:],11000)
            
                    

    elif test_type=='validate':
        if(len(video_dic.keys())==len(audio_dic.keys())):
            instruments = video_dic.keys()
        else:
            print("different number of instruments in video and audio!")
            print(len(video_dic.keys()))
            print(len(audio_dic.keys()))
            return
        for instru in instruments:
            if(len(video_dic[instru].keys())==len(audio_dic[instru].keys())):
                cases = video_dic[instru].keys()
            else:
                print("different number of cases in instrument " + str(instru)+'!')
                print(len(video_dic[instru].keys()))
                print(len(audio_dic[instru].keys()))
                return
            for case in cases:
                if(len(video_dic[instru][case])==len(audio_dic[instru][case[0:-4]+'.wav'])):
                    case_length = len(video_dic[instru][case])
                else:
                    print("different number of blocks in instrument " + str(instru)+" case"+str(case)+'!')
                    print(len(video_dic[instru][case]))
                    print(len(audio_dic[instru][case[0:-4]+'.wav']))
                    return
                destination1 = os.path.join(result_dir,str(instru) +'-'+str(case)+'-1.wav')
                destination2 = os.path.join(result_dir,str(instru) +'-'+str(case)+'-2.wav')
                wave1 = np.array([])
                wave2 = np.array([])
                wave_sets = []
                spec_sets = []
                # cluster by total data
                print(cases)
                origin_wave,_ =librosa.load(os.path.join(audio_dir,instru,case[0:-4]+'.wav'),sr=11000)
                for n in range(0,case_length):
                    sub_wave_set = []
                    sub_spec_set = []
                    stop_idx = 4
                    # cluster by block at stop_idx, after which the program will end up
                    
                    video_input = video_dic[instru][case][n]
                    audio_input = audio_dic[instru][case[0:-4]+'.wav'][n]
                    # the following part is to test the test part with certain data 
                    '''
                    video_input1 = np.load('D:\\huyb\\std\\video_3frames\\acoustic_guitar\\1\\3.npy')
                    video_input2 = np.load('D:\\huyb\\std\\video_3frames\\flute\\1\\3.npy')
                    cv2.imshow('a img',video_input1[0,:,:,:])
                    cv2.waitKey(0)
                    audio_input1 = np.load('D:\\huyb\\std\\audio_spectrums_linear\\flute\\1\\3.npy')
                    audio_input2 = np.load('D:\\huyb\\std\\audio_spectrums_linear\\acoustic_guitar\\1\\3.npy')
                    audio_input = audio_input1 + audio_input2
                    audio_input = audio_input[np.newaxis,np.newaxis,np.newaxis,:]
                    video_input = np.zeros(video_input1.shape,dtype = 'uint8')
                    gap = video_input1.shape[2]/2
                    for p in range(0,3):
                        video_input[p,:,0:112,:] = cv2.resize(video_input2[p, :, :, :], (112,224))
                        video_input[p,:,112:224,:] = cv2.resize(video_input1[p, :, :, :], (112,224))
                    video_input = np.transpose(video_input,(0,3,1,2))
                    video_input = video_input[np.newaxis,:]
                    
                    
                    temp = np.sum(np.stack([audio_input1,audio_input2]),axis = 0)
                    # temp = temp1 + temp2
                    temp = temp[np.newaxis,np.newaxis,np.newaxis,:]

                    audio_input = temp
                    '''
                    print(video_input.shape)
                    test_image = np.transpose(video_input,(0,1,3,4,2))
                    plt.imshow(test_image[0,0,:,:,:])
                    plt.show()

                    # print("video input",video_input.shape)
                    # print("audio input",audio_input.shape)
                    print(n,case_length)
                    if n==stop_idx:
                        np.save('audio_input',audio_input)
                        np.save('video_input_left',video_input)
                        exit()
                    estimated_spects = test1step(video_net,audio_net,syn_net,video_input,
                                                audio_input,
                                                device)      
                    if n == 0: 
                        temp = amplitude_to_db(np.absolute(audio_input),ref=np.max).transpose().flatten()
                        spec_sets.append(temp)   
                        sub_spec_set.append(temp)  
                        for idx1 in range(estimated_spects.shape[1]):
                            for idx2 in range(estimated_spects.shape[2]):
                                plt.figure()
                                plt.imshow(estimated_spects[0,idx1,idx2,0,:,:] * amplitude_to_db(np.absolute(audio_input[0,0,0,:,:]),ref=np.max))
                                plt.savefig('./left2-'+str(n)+'.jpg')
                                plt.show()
                                sub_wave_set.append(mask2wave(estimated_spects[0,idx1,idx2,0,:,:] * audio_input,'linear'))

                                wave_sets.append(mask2wave(estimated_spects[0,idx1,idx2,0,:,:] * audio_input,'linear'))
                                
                                temp = amplitude_to_db(np.absolute(estimated_spects[0,idx1,idx2,0,:,:] * audio_input),ref = np.max)
                                temp = temp.transpose().flatten()
                                spec_sets.append(temp)
                                sub_spec_set.append(temp)
                    else:
                        temp = amplitude_to_db(np.absolute(audio_input),ref=np.max).transpose().flatten()
                        spec_sets[0] = np.append(spec_sets[0], temp)
                        sub_spec_set.append(temp)
                        for idx1 in range(estimated_spects.shape[1]):
                            for idx2 in range(estimated_spects.shape[2]):
                                plt.figure()
                                plt.imshow(estimated_spects[0,idx1,idx2,0,:,:] * amplitude_to_db(np.absolute(audio_input[0,0,0,:,:]),ref=np.max))
                                plt.savefig('./left2-'+str(n)+'.jpg')
                                plt.show()
                                if n==3:
                                    exit()

                                sub_wave_set.append(mask2wave(estimated_spects[0,idx1,idx2,0,:,:] * audio_input,'linear'))

                                wave_sets[idx1*estimated_spects.shape[1]+idx2] = np.append(
                                                wave_sets[idx1*estimated_spects.shape[1]+idx2],
                                                mask2wave(estimated_spects[0,idx1,idx2,0,:,:] * audio_input,'linear'))
                                
                                temp = amplitude_to_db(np.absolute(estimated_spects[0,idx1,idx2,0,:,:] * audio_input),ref = np.max)
                                temp = temp.transpose().flatten()
                                spec_sets[idx1*estimated_spects.shape[1]+idx2+1] = np.append(
                                                spec_sets[idx1*estimated_spects.shape[1]+idx2+1],
                                                temp.reshape(1,-1))
                                sub_spec_set.append(temp)            
                    
                    sub_wave_set = np.array(sub_wave_set)
                    print("subwave",sub_wave_set.shape)
                    sub_wave_set = np.insert(sub_wave_set,0,mask2wave(audio_input,'linear'),0)
                    print("subwave",sub_wave_set.shape)
                    # cluster with sub wave
                    # 
                    '''
                    if n==4: 
                        clust_estimator.fit(sub_wave_set)
                        for idx1 in range(estimated_spects.shape[1]):
                            for idx2 in range(estimated_spects.shape[2]):
                                write_wav(os.path.join('./testsub',str(idx1)+'-'+str(idx2)+'.wav'),sub_wave_set[1+idx1*estimated_spects.shape[1]+idx2,:],11000)
                        
                        labels = clust_estimator.labels_
                        print(labels)

                        with open('./testsub/label.txt','w') as file:
                            file.write(str(labels[0]))
                            file.write('\n')
                            for idx1 in range(estimated_spects.shape[1]):
                                file.write(str(labels[1+idx1*estimated_spects.shape[2]:(idx1+1)*estimated_spects.shape[2]+1]))
                                file.write('\n')
                        exit()
                    '''
                    # cluster with sub spect
                    #
                    '''
                    sub_spec_set = np.array(sub_spec_set)
                    print("subspec",sub_spec_set.shape)
                    if n==0: 
                        clust_estimator.fit(sub_spec_set)
                        for idx1 in range(estimated_spects.shape[1]):
                            for idx2 in range(estimated_spects.shape[2]):
                                write_wav(os.path.join('./testsub',str(idx1)+'-'+str(idx2)+'.wav'),sub_wave_set[1+idx1*estimated_spects.shape[1]+idx2,:],11000)
                        
                        labels = clust_estimator.labels_
                        print(labels)

                        with open('./testsub/label.txt','w') as file:
                            file.write(str(labels[0]))
                            file.write('\n')
                            for idx1 in range(estimated_spects.shape[1]):
                                file.write(str(labels[1+idx1*estimated_spects.shape[2]:(idx1+1)*estimated_spects.shape[2]+1]))
                                file.write('\n')
                        exit()
                    '''
                # cluster with total data
                # 
                wave_sets = np.array(wave_sets)
                print("waveset",wave_sets.shape)
                BLOCK_LENGTH = 66302
                length = math.floor(len(origin_wave)/BLOCK_LENGTH)*BLOCK_LENGTH
                wave_sets = np.insert(wave_sets,0,origin_wave[0:length],0)
                print("waveset",wave_sets.shape)

                spec_sets = np.array(spec_sets)
                print("specset",spec_sets.shape)

                # clust_estimator.fit(wave_sets)
                # clust_estimator.fit(spec_sets)
                
                for idx1 in range(estimated_spects.shape[1]):
                    for idx2 in range(estimated_spects.shape[2]):
                        write_wav(os.path.join('./test',str(idx1)+'-'+str(idx2)+'.wav'),wave_sets[1+idx1*estimated_spects.shape[1]+idx2,:],11000) 
                
                # labels = clust_estimator.labels_

                # print(labels)
                '''    
                with open('label.txt','w') as file:
                    for idx1 in range(estimated_spects.shape[1]):
                        file.write(str(labels[idx1*estimated_spects.shape[2]:(idx1+1)*estimated_spects.shape[2]-1]))
                        file.write('\n')
                '''
                exit()


def test_evaluate(ground_truth_dir,test_result_dir):
    ground_truth_list = os.listdir(ground_truth_dir)
    test_result_list = os.listdir(test_result_dir)
    total_sdr = 0
    for i in range(len(ground_truth_list)):
        # print(os.path.join(test_result_dir,test_result_list[i]))
        test_result,_ = librosa.load(os.path.join(test_result_dir,test_result_list[i]),sr=11000)
        test_result = librosa.resample(test_result,11000,44100)
        ground_truth,_ = librosa.load(os.path.join(ground_truth_dir,ground_truth_list[i]),sr=44100)
        test_result = np.append(test_result,ground_truth[len(test_result):])
        # print(test_result)
        # print(np.array(test_result).shape,np.array(ground_truth).shape)
    
        test_spect = librosa.stft(test_result)   
        ground_spect = librosa.stft(ground_truth) 
        '''
        plt.subplot(1,2,1)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(test_spect),ref=np.max),y_axis='log', x_axis='time') 
        plt.subplot(1,2,2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(ground_spect),ref=np.max),y_axis='log', x_axis='time')
        plt.show()
        '''
        [sdr, _, _, _] = mir_eval.separation.bss_eval_sources(test_result,ground_truth)
        print(test_result_list[i],ground_truth_list[i],sdr)

        total_sdr += sdr
        '''
        [sdr, _, _, _] = mir_eval.separation.bss_eval_sources(ground_truth,ground_truth)
        print(ground_truth_list[i],ground_truth_list[i],sdr)
        '''
    print(total_sdr/len(ground_truth_list))

if __name__ == '__main__':
    test_train1step()
