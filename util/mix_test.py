import sys
sys.path.append('..')
from models.models import modifyresnet18, UNet7, synthesizer
from util import waveoperate
from util.datahelper import image_normalization

import wave
import os
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

NEW_DIR = './audio_spectrums'
WAVE_DIR = '/media/huyb/EB9C554C4A073FFC/课程用/大三上/视听导/dataset/dataset/audios/solo'
SPEC_DIR = '../../audio_spectrums_linear' 
VIDEO_DIR = '../../video_3frames'
MODEL_DIR = '../../model'
SAMPLE_RATE = 11000
BLOCK_LENGTH = 66303
WINDOW_SIZE = 1022
HOP_LENGTH = 256

LOAD_WAVE = False
LOAD_SPEC = bool(1-LOAD_WAVE)

MODE = "test"

frequencies = np.linspace(SAMPLE_RATE/2/512,SAMPLE_RATE/2,512)
log_freq = np.log10(frequencies)
sample_freq = np.linspace(log_freq[0],log_freq[-1],256)
sample_index = [np.abs(log_freq-x).argmin() for x in sample_freq]
# prepare for log resample
sample_index2 = np.linspace(0,510,256)
sample_index2 = sample_index2.astype(int)
#print(sample_index2.dtype)
#exit()
i = 1
###############################################################################3
# load wave data
if LOAD_WAVE:
	instrument_class = os.listdir(WAVE_DIR)
	instrument1 = instrument_class[4]
	instrument2 = instrument_class[0]

	wave_dir1 = os.listdir(os.path.join(WAVE_DIR,instrument1))[1]
	wave_dir2 = os.listdir(os.path.join(WAVE_DIR,instrument2))[1]

	print("load music from",os.path.join(WAVE_DIR,instrument2,wave_dir2),os.path.join(WAVE_DIR,instrument1,wave_dir1))
	w,sr = librosa.load(os.path.join(WAVE_DIR,instrument1,wave_dir1),sr=SAMPLE_RATE)
	data1 = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH-1]

	w,sr = librosa.load(os.path.join(WAVE_DIR,instrument2,wave_dir2),sr=SAMPLE_RATE)
	data2 = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH-1]

	data = data1 + data2
	# librosa.output.write_wav(os.path.join('./test','mix.wav'),data,sr) 
	# librosa.output.write_wav(os.path.join('./test','src1.wav'),data1,sr) 
	# librosa.output.write_wav(os.path.join('./test','src2.wav'),data2,sr) 

	stft_data = librosa.stft(data,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
	stft_data1 = librosa.stft(data1,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
	stft_data2 = librosa.stft(data2,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
	print("length0",data1.shape)

	stft_data = stft_data1[sample_index2,:] 
	stft_data1 = stft_data1[sample_index2,:]
	stft_data2 = stft_data2[sample_index2,:]

	abs_stft_data1 = np.abs(stft_data1)
	abs_stft_data2 = np.abs(stft_data2)
	abs_stft_data = abs_stft_data1 + abs_stft_data2

################################################################

################################################################
# load from spec
if LOAD_SPEC:
	instrument_class = os.listdir(SPEC_DIR)
	instrument1 = instrument_class[4]
	instrument2 = instrument_class[3]

	spec_dir1 = os.listdir(os.path.join(SPEC_DIR,instrument1))[1]
	spec_dir2 = os.listdir(os.path.join(SPEC_DIR,instrument2))[1]

	stft_data1 = np.load(os.path.join(SPEC_DIR,instrument1,spec_dir1,str(i)+'.npy'))
	stft_data2 = np.load(os.path.join(SPEC_DIR,instrument2,spec_dir2,str(i)+'.npy'))

	abs_stft_data1 = librosa.amplitude_to_db(np.abs(stft_data1),ref=np.max)
	abs_stft_data2 = librosa.amplitude_to_db(np.abs(stft_data2),ref=np.max)
	stft_data = (stft_data1 + stft_data2)
	abs_stft_data = abs_stft_data1 + abs_stft_data2

	print("load spectrum from",os.path.join(SPEC_DIR,instrument2,spec_dir2),os.path.join(SPEC_DIR,instrument1,spec_dir1))


################################################################

################################################################
# load video data
frame_dir1 = os.listdir(os.path.join(VIDEO_DIR,instrument1))[0]
frame_dir2 = os.listdir(os.path.join(VIDEO_DIR,instrument2))[0]

frame_data1 = np.load(os.path.join(VIDEO_DIR,instrument1,frame_dir1,str(i)+'.npy'))
frame_data2 = np.load(os.path.join(VIDEO_DIR,instrument2,frame_dir2,str(i)+'.npy'))
print("load frame from",os.path.join(VIDEO_DIR,instrument2,frame_dir2),os.path.join(VIDEO_DIR,instrument1,frame_dir1))

frame_data1 = np.transpose(frame_data1,(0,3,1,2))
frame_data2 = np.transpose(frame_data2,(0,3,1,2))
#################################################################

#################################################################
# prepare for network
abs_stft_data = abs_stft_data[np.newaxis,np.newaxis,:]
frame_data1 = frame_data1[np.newaxis,:]
frame_data2 = frame_data2[np.newaxis,:]
#################################################################

# mask = abs_stft_data1 >= abs_stft_data2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_net = modifyresnet18(1).to(device)
audio_net = UNet7().to(device)
syn_net = synthesizer().to(device)

video_net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'video_net_params.pkl')))
audio_net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'audio_net_params.pkl')))
syn_net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'syn_net_params.pkl')))

audio_input = torch.from_numpy(abs_stft_data).to(device)
audio_out = audio_net.forward(audio_input)

if MODE != "test":	
	video_input = image_normalization(frame_data1).to(device)	
	video_out = video_net.forward(video_input[0,:,:,:,:])
	temp = video_out * audio_out
	temp = torch.transpose(temp, 1, 2)
	temp = torch.transpose(temp, 2, 3)
	syn_act = syn_net.forward(temp)
	syn_act = torch.transpose(syn_act, 2, 3)
	syn_act = torch.transpose(syn_act, 1, 2)

	mask = syn_act[0,0,:,:].cpu().detach().numpy()
	print(mask[0:10,0])

	result1 = np.zeros(stft_data.shape,dtype = complex)
	result1 = stft_data * mask

	video_input = image_normalization(frame_data2).to(device)
	video_out = video_net.forward(video_input[0,:,:,:,:])
	print(video_out.shape)
	temp = video_out * audio_out
	temp = torch.transpose(temp, 1, 2)
	temp = torch.transpose(temp, 2, 3)
	syn_act = syn_net.forward(temp)
	syn_act = torch.transpose(syn_act, 2, 3)
	syn_act = torch.transpose(syn_act, 1, 2)

	mask = syn_act[0,0,:,:].cpu().detach().numpy()
	print(mask[0:10,0])

	result2 = np.zeros(stft_data.shape,dtype = complex)
	result2 = stft_data * mask

	plt.subplot(1,3,1)
	librosa.display.specshow((np.abs(stft_data)),y_axis='linear', x_axis='time')
	plt.subplot(1,3,2)
	librosa.display.specshow((np.abs(result1)),y_axis='linear', x_axis='time')
	plt.subplot(1,3,3)
	librosa.display.specshow((np.abs(result2)),y_axis='linear', x_axis='time')
	plt.show()


	# result_data1 = librosa.istft(result1,hop_length = HOP_LENGTH,win_length=1022)
	# result_data2 = librosa.istft(result2,hop_length = HOP_LENGTH,win_length=1022)

	mix_wave = waveoperate.mask2wave(stft_data,'linear')
	result_data1 = waveoperate.mask2wave(result1,'linear')
	result_data2 = waveoperate.mask2wave(result2,'linear')

	librosa.output.write_wav(os.path.join('./test','mix.wav'),result_data1,SAMPLE_RATE) 
	librosa.output.write_wav(os.path.join('./test','result1.wav'),result_data1,SAMPLE_RATE) 
	librosa.output.write_wav(os.path.join('./test','result2.wav'),result_data2,SAMPLE_RATE) 
else:
	video_input = image_normalization(frame_data1).to(device)
	video_out = video_net.forward(video_input[0,:,:,:,:],mode=MODE)
	layer = nn.MaxPool2d(kernel_size = 14)
	# video_out = layer(video_out)
	column = 1
	row = 1
	column_init = 0
	row_init = 0

	# for kk in range(0,16):
	# 	print(np.mean(video_out[0,kk,:,:].cpu().detach().numpy()))

	for m in range(row_init,row_init+row):
		for n in range(column_init,column_init+column):
			video_select = video_out[:,:,m,n]
			video_select = video_select[:,np.newaxis,np.newaxis]
			video_select = torch.transpose(video_select,0,2)
			video_select = torch.transpose(video_select,1,3)
			print(video_select.shape,audio_out.shape)
			temp = video_select * audio_out
			temp = torch.transpose(temp, 1, 2)
			temp = torch.transpose(temp, 2, 3)
			syn_act = syn_net.forward(temp)
			syn_act = torch.transpose(syn_act, 2, 3)
			syn_act = torch.transpose(syn_act, 1, 2)

			mask = syn_act[0,0,:,:].cpu().detach().numpy()
			print(mask[0:10,0])

			stft_data = stft_data1 + stft_data2
			result = stft_data * mask
			result_data = waveoperate.mask2wave(result,'linear')
			librosa.output.write_wav(os.path.join('./test','result1.wav'),result_data,SAMPLE_RATE) 

			# plt.subplot(row,column,(m-row_init)*row+(n-column_init)+1)
			plt.subplot(row,column,0+(m-row_init)*row+(n-column_init)+1)
			result = librosa.amplitude_to_db(np.abs(result),ref = np.max)
			librosa.display.specshow(result,y_axis='linear', x_axis='time')

	plt.show()


	video_input = image_normalization(frame_data2).to(device)
	video_out = video_net.forward(video_input[0,:,:,:,:],mode=MODE)
	layer = nn.MaxPool2d(kernel_size = 14)
	# video_out = layer(video_out)

	# for kk in range(0,16):
	# 	print(np.mean(video_out[0,kk,:,:].cpu().detach().numpy()))

	plt.figure()
	for m in range(row_init,row_init+row):
		for n in range(column_init,column_init+column):
			video_select = video_out[:,:,m,n]
			video_select = video_select[:,np.newaxis,np.newaxis]
			video_select = torch.transpose(video_select,0,2)
			video_select = torch.transpose(video_select,1,3)
			print(video_select.shape,audio_out.shape)
			temp = video_select * audio_out
			temp = torch.transpose(temp, 1, 2)
			temp = torch.transpose(temp, 2, 3)
			syn_act = syn_net.forward(temp)
			syn_act = torch.transpose(syn_act, 2, 3)
			syn_act = torch.transpose(syn_act, 1, 2)

			mask = syn_act[0,0,:,:].cpu().detach().numpy()
			print(mask[0:10,0])

			stft_data = stft_data1 + stft_data2
			result = stft_data * mask

			result_data = waveoperate.mask2wave(result,'linear')
			librosa.output.write_wav(os.path.join('./test','result2.wav'),result_data,SAMPLE_RATE)

			# plt.subplot(row,column,(m-row_init)*row+(n-column_init)+1)
			plt.subplot(row,column,(m-row_init)*row+(n-column_init)+1)
			result = librosa.amplitude_to_db(np.abs(result),ref = np.max)
			librosa.display.specshow(result,y_axis='linear', x_axis='time')

	plt.show()


