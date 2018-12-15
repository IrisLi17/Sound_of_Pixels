import os
import cv2
from pydub import AudioSegment
import numpy as np
import wave
from random import randint, sample
import torchvision.transforms as transforms
import torch
from librosa import amplitude_to_db
import math


def convert_video(src, dst):
    # dst string cannot be src string
    cmd_str = 'ffmpeg -i ' + src + ' -strict -2 -qscale 0 -r 24 -ar 44100 -ac 1 -y ' + dst
    os.system(cmd_str)


def read_video_fordir(dirname, savedir1, savedir, fre):
    if os.path.exists(savedir1) == False:
        os.mkdir(savedir1)
    vidlist = os.listdir(dirname)
    if os.path.exists(savedir) == False:
        os.mkdir(savedir)
    '''
    #convert videos to .mp4 file with rate=24, audio rate=44100 and audio channel=1
    vidlist=os.listdir(dirname)
    for vid in vidlist:
        src=os.path.join(dirname,vid)
        dst=os.path.join(dirname,new_name+'.mp4')
        cmd_str='ffmpeg -i '+dirname+'/'+vid+' -strict -2 -qscale 0 -r 24 -ar 44100 -ac 1 -y '+dirname+'/'+new_name+'.mp4'
        os.system(cmd_str)
        cmd_str='rm '+dirname+'/'+vid
        os.system(cmd_str)
    '''
    vidlist = os.listdir(dirname)
    for vid in vidlist:
        name1 = os.path.splitext(vid)[0]
        name2 = os.path.splitext(vid)[-1][1:]
        song = AudioSegment.from_file(os.path.join(dirname, vid), name2)
        song.export(os.path.join(savedir1, name1 + '.wav'), format='wav')
        if os.path.exists(os.path.join(savedir, name1)) == False:
            os.mkdir(os.path.join(savedir, name1))
        cap = cv2.VideoCapture(os.path.join(dirname, vid))
        c = 1
        d = 1
        while (cap.isOpened()):
            ret, frame = cap.read()
            if (ret == 0):
                break
            if (c % fre == 0):
                cv2.imwrite(os.path.join(savedir, name1, "%06d" %
                                         d + '.jpg'), frame)
                d = d + 1
            c = c + 1
            k = cv2.waitKey(20)
        cap.release()


def read_video(dirname, savedir1, savedir, fre, vidno):
    if os.path.exists(savedir1) == False:
        os.mkdir(savedir1)
    vidlist = os.listdir(dirname)
    if os.path.exists(savedir) == False:
        os.mkdir(savedir)
    vidlist = os.listdir(dirname)
    for vid in vidlist:
        name1 = os.path.splitext(vid)[0]
        name2 = os.path.splitext(vid)[-1][1:]
        print(name1)
        if int(name1) == vidno:
            print(os.path.join(dirname, vid))
            song = AudioSegment.from_file(os.path.join(dirname, vid), name2)
            song.export(os.path.join(savedir1, name1 + '.wav'), format='wav')
            if os.path.exists(os.path.join(savedir, name1)) == False:
                os.mkdir(os.path.join(savedir, name1))
            cap = cv2.VideoCapture(os.path.join(dirname, vid))
            c = 1
            d = 1
            while (cap.isOpened()):
                ret, frame = cap.read()
                if (ret == 0):
                    break
                if (c % fre == 0):
                    cv2.imwrite(os.path.join(savedir, name1,
                                             "%06d" % d + '.jpg'), frame)
                    d = d + 1
                c = c + 1
                k = cv2.waitKey(20)
            cap.release()


# read_video_fordir('videos/test/duet/fluteviolin','audios/test/duet/fluteviolin','images/test/duet/fluteviolin',10)
def read_audio(audio_file):
    f = wave.open(audio_file, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData, dtype=np.int16)
    # normalize
    waveData = waveData * 1.0 / (max(abs(waveData)))
    time = np.arange(0, nframes) * (1.0 / framerate)
    return waveData


def audio_video_sim(audiodata, videodata):
    print('calculate similarity')
    audio_T = audiodata.transpose()
    audio_sample = sample(audio_T, videodata.shape[0])
    audio_sample = np.squeeze(audio_sample)
    w, h = videodata.shape[1], videodata.shape[2]
    channel = 1
    sim_list = []
    print
    w, h
    for i in range(w):
        sim_list.append([])
        for j in range(h):
            pixeldata = videodata[:, i, j, channel]
            pixeldata = np.squeeze(pixeldata)
            # print pixeldata.shape
            # print audio_sample.shape
            sim_ratio = dist_sim(pixeldata, audio_sample)
            # print sim_ratio
            sim_list[i].append(sim_ratio)

    sim_martrx = np.asarray(sim_list)
    # print sim_martrx.shape
    pos = np.where(sim_martrx == np.max(sim_martrx))
    return pos


def audio_video_sim2(audiodata, audiodata2, videodata):
    '''
    param:audiodata np.array (1,samplelength)
    param:audiodata2 np.array (1,samplelength)
    param:videodata np.array  (num,w,h,channel)
    '''
    print('calculate similarity')
    audio_T = audiodata.transpose()
    audio_sample = sample(audio_T, videodata.shape[0])
    audio_sample = np.squeeze(audio_sample)

    audio_T = audiodata2.transpose()
    audio_sample_2 = sample(audio_T, videodata.shape[0])
    audio_sample_2 = np.squeeze(audio_sample_2)

    w, h = videodata.shape[1], videodata.shape[2]
    channel = 1
    sim_list = []
    sim_list_2 = []
    print
    w, h
    for i in range(w):
        sim_list.append([])
        sim_list_2.append([])
        for j in range(h):
            pixeldata = videodata[:, i, j, channel]
            pixeldata = np.squeeze(pixeldata)
            # print pixeldata.shape
            # print audio_sample.shape
            sim_ratio = dist_sim(pixeldata, audio_sample)
            sim_ratio_2 = dist_sim(pixeldata, audio_sample_2)
            # print sim_ratio
            sim_list[i].append(sim_ratio)

            sim_list_2[i].append(sim_ratio_2)

    sim_martrx = np.asarray(sim_list)
    # print sim_martrx.shape
    pos_1 = np.where(sim_martrx == np.max(sim_martrx))

    sim_martrx = np.asarray(sim_list_2)

    pos_2 = np.where(sim_martrx == np.max(sim_martrx))

    if pos_1[0] > pos_2[0]:
        return [0, 1]
    else:
        return [1, 0]

    return [0, 1]


def dist_sim(A, B):
    dist = np.linalg.norm(A - B)
    sim = 1.0 / (1.0 + dist)
    return sim


def load_data(image_data_dir, audio_datafile, hasnpy=False, npypath=""):
    if hasnpy:
        npvideos, npaudios = np.load(npypath)
        return npvideos, npaudios

    videos = []
    audios = []
    label_dirs = os.listdir(image_data_dir)
    label_dirs.sort()
    for _label_dir in label_dirs:
        print
        'loaded {}'.format(_label_dir)
        imgs_name = os.listdir(os.path.join(image_data_dir, _label_dir))
        imgs_name.sort()
        for img_name in imgs_name:
            im_ar = cv2.imread(os.path.join(
                image_data_dir, _label_dir, img_name))
            im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
            im_ar = np.asarray(im_ar)
            im_ar = preprocess(im_ar)
            videos.append(im_ar)
        audio_data = read_audio(audio_datafile)
        audios.append(audio_data)
    npvideos = np.array(videos)
    npaudios = np.array(audios)
    if not os.path.exists("data.npy"):
        np.save("data.npy", (npvideos, npaudios))
    return npvideos, npaudios


def load_data_from_image_file(image_data_dir, audio_datafile):
    videos = []
    audios = []
    label_dirs = os.listdir(image_data_dir)
    label_dirs.sort()
    for img_name in label_dirs:
        im_ar = cv2.imread(os.path.join(image_data_dir, img_name))
        im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
        im_ar = np.asarray(im_ar)
        im_ar = preprocess(im_ar)
        videos.append(im_ar)
    audio_data = read_audio(audio_datafile)
    audios.append(audio_data)
    npvideos = np.array(videos)
    npaudios = np.array(audios)
    return npvideos, npaudios


def preprocess(im_ar):
    im_ar = cv2.resize(im_ar, (224, 224))
    im_ar = im_ar / 255.0
    return im_ar


def load_data_from_video(VideoName, audio_datafile, frequency):
    videos = []
    audios = []
    cap = cv2.VideoCapture(VideoName)
    c = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret == 0):
            break
        if (c % frequency == 0):
            im_ar = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_ar = np.asarray(im_ar)
            im_ar = preprocess(im_ar)
            videos.append(im_ar)
        c = c + 1
    cap.release()
    audio_data = read_audio(audio_datafile)
    audios.append(audio_data)
    npvideos = np.array(videos)
    npaudios = np.array(audios)
    return npvideos, npaudios


def load_test_data(videodir, adiodir, BLOCK_LENGTH = 66302, WINDOW_SIZE = 1022,
                    HOP_LENGTH = 256, SAMPLE_RATE=11000, FPS=24, fstype='linear'):
    """
    :videodir: the location where video file locates
    :audiodir: the location where audio file locates
    :BLOCK_LENGTH: the number of sample points a block of wave have
                   default 66302 makes the block about 6s under sr of 11kHz
    :SAMPLE_RATE: sample rate of audio 
    :WINDOW_SIZE: window size of stft
    :HOP_LENGTH: hop length of stft
    :FPS: frame per second of video
    :fstype: the type when sample frequencies after stft
             accept 'linear' or 'log'
    """
    # batchsize = 1
    video_data = {}
    audio_data = {}
    BLOCK_TIME = BLOCK_LENGTH/SAMPLE_RATE
    BLOCK_LENGTH = math.floor(BLOCK_TIME*FPS)
    FRAME_INDEX = [0, math.floor((BLOCK_LENGTH-1)/2), BLOCK_LENGTH-1]
    instruments = os.listdir(videodir)
    for instru in instruments:
        video_data[instru] = {}
        instru_dir = os.path.join(videodir, instru)
        cases = os.listdir(instru_dir)
        for case in cases:
            video_data[instru][case] = []
            case_dir = os.path.join(instru_dir, case)
            items = os.listdir(case_dir)
            for item in items:
                # temp = amplitude_to_db(np.absolute(np.load(os.path.join(case_dir, item))),ref=np.max)
                temp = np.load(os.path.join(case_dir, item))
                cap = cv2.VideoCapture(videodir)
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                block_num = math.floor(frameCount/BLOCK_LENGTH)
                buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
                fc = 0
                ret = True
                while (fc < frameCount and ret):
                    ret, buf[fc] = cap.read()
                    fc += 1
                cap.release()
                for i in range(block_num):
                    temp = buf[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH, :, :, :]
                    result = temp[FRAME_INDEX, :, :, :]
                    final = np.empty((len(FRAME_INDEX), 224, 224, 3),np.dtype('uint8'))
                    for p in range(0, len(FRAME_INDEX)):
                        final[p, :, :, :] = cv2.resize(result[p, :, :, :], (224, 224))
                        final = np.transpose(final,(0,3,1,2))
                        final = final[np.newaxis,:]
                        video_data[instru][case].append(final)

    frequencies = np.linspace(SAMPLE_RATE/2/512,SAMPLE_RATE/2,512)
    log_freq = np.log10(frequencies)
    ample_freq = np.linspace(log_freq[0],log_freq[-1],256)
    sample_index = np.array([np.abs(log_freq-x).argmin() for x in sample_freq])
    # prepare for log resample
    sample_index2 =np.array(np.linspace(0,510,256)).astype(int)

    instruments = os.listdir(audiodir)
    for instru in instruments:
        audio_data[instru] = {}
        instru_dir = os.path.join(audiodir, instru)
        cases = os.listdir(instru_dir)
        for case in cases:
            audio_data[instru][case] = []
            case_dir = os.path.join(instru_dir, case)
            items = os.listdir(case_dir)
            for item in items:
                w,_ = librosa.load(audiodir,sr=SAMPLE_RATE)
                block_num = math.floor(len(w)/BLOCK_LENGTH)
                for i in range(block_num):
                    # select the wave data
                    data = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH]
                    # STFT
                    stft_data = librosa.stft(data,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
                    # log scale resample
                    if fstype == 'log':
                        stft_data = stft_data[sample_index,:]
                    elif fstype == 'linear':
                        stft_data = stft_data[sample_index2,:]
                    # print(stft_data.shape)
                    # save data
                    stft_data = stft_data[np,newaxis,np.newaxis,np.newaxis, :]
                    audio_data[instru][case].append(stft_data)

    return [video_data,audio_data]

    


def load_all_training_data(spec_dir, image_dir):
    spec_data = {}
    instruments = os.listdir(spec_dir)
    for instru in instruments:
        spec_data[instru] = {}
        instru_dir = os.path.join(spec_dir, instru)
        cases = os.listdir(instru_dir)
        for case in cases:
            spec_data[instru][case] = []
            case_dir = os.path.join(instru_dir, case)
            items = os.listdir(case_dir)
            for item in items:
                # temp = amplitude_to_db(np.absolute(np.load(os.path.join(case_dir, item))),ref=np.max)
                temp = np.load(os.path.join(case_dir, item))
                temp = temp[np.newaxis, :]
                spec_data[instru][case].append(temp)
    image_data = {}
    instruments = os.listdir(image_dir)
    for instru in instruments:
        image_data[instru] = {}
        instru_dir = os.path.join(image_dir, instru)
        cases = os.listdir(instru_dir)
        for case in cases:
            image_data[instru][case] = []
            case_dir = os.path.join(instru_dir, case)
            items = os.listdir(case_dir)
            for item in items:
                image_data[instru][case].append(np.transpose(
                    np.load(os.path.join(case_dir, item)), (0, 3, 1, 2)))
    print(spec_data['accordion']['1'][0].shape)
    print(image_data['accordion']['1'][0].shape)
    return [spec_data, image_data]


def sample_from_dict(spec_data, image_data, N=2):
    sampled_spec = []
    sampled_image = []
    instru_idx = sample(spec_data.keys(), N)
    for instru in instru_idx:
        case_idx = sample(spec_data[instru].keys(), 1)
        case = case_idx[0]
        item_idx = sample(
            range(min(len(spec_data[instru][case]), len(image_data[instru][case]))), 1)
        item = item_idx[0]
        sampled_spec.append(spec_data[instru][case][item])
        sampled_image.append(image_data[instru][case][item])
    sampled_spec = np.stack(sampled_spec, axis=0)
    sampled_image = np.stack(sampled_image, axis=0)
    # print(sampled_spec.shape)
    # print(sampled_image.shape)
    return [sampled_spec, sampled_image]


def sample_input(spec_dir, image_dir, phase, N=2, fraction=0.7):
    assert len(os.listdir(spec_dir)) == len(os.listdir(image_dir))
    selected_instruments = sample(os.listdir(spec_dir), N)
    selected_spec_dirs = [os.path.join(spec_dir, i)
                          for i in selected_instruments]
    selected_image_dirs = [os.path.join(image_dir, i)
                           for i in selected_instruments]
    assert (len(os.listdir(selected_spec_dirs[i])) == len(
        os.listdir(selected_image_dirs[i])) for i in range(N))
    num_cases = [len(os.listdir(selected_spec_dirs[i])) for i in range(N)]
    if phase == 'train':
        selected_cases = [sample(os.listdir(selected_spec_dirs[i])[0:int(fraction * num_cases[i])], 1) for i in
                          range(N)]  # len: N
    elif phase == 'validate':
        selected_cases = [sample(os.listdir(selected_spec_dirs[i])[int(fraction * num_cases[i]):], 1) for i in
                          range(N)]  # len: N
    # print(selected_cases)
    spec_cases_dirs = [os.path.join(
        selected_spec_dirs[i], selected_cases[i][0]) for i in range(N)]
    image_cases_dirs = [os.path.join(
        selected_image_dirs[i], selected_cases[i][0]) for i in range(N)]
    assert (len(os.listdir(spec_cases_dirs[i])) == len(
        os.listdir(image_cases_dirs[i])) for i in range(N))
    selected_frames = [sample(os.listdir(spec_cases_dirs[i]), 1)
                       for i in range(N)]
    spec_frames_dirs = [os.path.join(
        spec_cases_dirs[i], selected_frames[i][0]) for i in range(N)]
    image_frames_dirs = [os.path.join(
        image_cases_dirs[i], selected_frames[i][0]) for i in range(N)]
    try:
        spect_input = [np.absolute(np.load(spec_frames_dirs[i]))
                       for i in range(N)]
        for i in range(N):
            spect_input[i] = spect_input[i][np.newaxis, :]
        spect_input = np.stack([i for i in spect_input], axis=0)
        image_input = np.stack([np.transpose(
            np.load(image_frames_dirs[i]), (0, 3, 1, 2)) for i in range(N)], axis=0)
    except:
        return [None, None]
    return [spect_input, image_input]


def image_normalization(image_input):
    """
    :param image_input: numpy array of size (N, num_of_frames, number_of_channels, height, width), which is (N, 3, 3, 224, 224)
    :return:
    """
    normalize = torch.zeros(image_input.shape, dtype=torch.float32)
    image_input = image_input / 255.0
    # print('image input' + str(image_input))
    for i in range(image_input.shape[0]):
        for frame in range(image_input.shape[1]):
            normalize[i, frame, :, :, :] = transforms.functional.normalize(
                torch.from_numpy(image_input[i, frame, :, :, :]),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    # print('normalized' + str(normalize))
    return normalize
