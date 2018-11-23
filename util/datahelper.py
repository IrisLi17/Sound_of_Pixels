import os
import cv2
from pydub import AudioSegment
import numpy as np
import wave
from random import randint, sample


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
                cv2.imwrite(os.path.join(savedir, name1, "%06d" % d + '.jpg'), frame)
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
                    cv2.imwrite(os.path.join(savedir, name1, "%06d" % d + '.jpg'), frame)
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
            im_ar = cv2.imread(os.path.join(image_data_dir, _label_dir, img_name))
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
