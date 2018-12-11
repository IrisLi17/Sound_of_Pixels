from models.models import modifyresnet18, UNet7, synthesizer
from util import waveoperate

import wave
import os
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

NEW_DIR = './audio_spectrums'
SPEC_DIR = '/media/huyb/EB9C554C4A073FFC/课程用/大三上/视听导/dataset/dataset/audios/solo' 
VIDEO_DIR = '/media/huyb/EB9C554C4A073FFC/课程用/大三上/视听导/dataset/dataset/audios/solo'
MODEL_DIR = '../model'
SAMPLE_RATE = 11000
BLOCK_LENGTH = 66303
WINDOW_SIZE = 1022
HOP_LENGTH = 256

###############################################################################3
# load wave data
frequencies = np.linspace(SAMPLE_RATE/2/512,SAMPLE_RATE/2,512)
log_freq = np.log10(frequencies)
sample_freq = np.linspace(log_freq[0],log_freq[-1],256)
sample_index = [np.abs(log_freq-x).argmin() for x in sample_freq]
# prepare for log resample
sample_index2 = np.linspace(0,510,256)
sample_index2 = sample_index2.astype(int)
#print(sample_index2.dtype)
#exit()

instrument_class = os.listdir(SPEC_DIR)
instrument1 = instrument_class[4]
instrument2 = instrument_class[1]

files_dir1 = os.path.join(SPEC_DIR,instrument1)
files_dir2 = os.path.join(SPEC_DIR,instrument2)

file_dir1 = os.listdir(files_dir1)[1]
file_dir2 = os.listdir(files_dir2)[1]

i = 1
print("load music from",os.path.join(files_dir2,file_dir2),os.path.join(files_dir1,file_dir1))
w,sr = librosa.load(os.path.join(files_dir1,file_dir1),sr=SAMPLE_RATE)
data1 = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH-1]

w,sr = librosa.load(os.path.join(files_dir2,file_dir2),sr=SAMPLE_RATE)
data2 = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH-1]

data = data1 + data2
# librosa.output.write_wav(os.path.join('./test','mix.wav'),data,sr) 
# librosa.output.write_wav(os.path.join('./test','src1.wav'),data1,sr) 
# librosa.output.write_wav(os.path.join('./test','src2.wav'),data2,sr) 

stft_data = librosa.stft(data,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
stft_data1 = librosa.stft(data1,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
stft_data2 = librosa.stft(data2,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
print("length0",data1.shape)

stft_data = stft_data[sample_index2,:]
stft_data1 = stft_data1[sample_index2,:]
stft_data2 = stft_data2[sample_index2,:]

abs_stft_data = np.abs(stft_data)
abs_stft_data1 = np.abs(stft_data1)
abs_stft_data2 = np.abs(stft_data2)

################################################################

################################################################
# load video data

mask = abs_stft_data1 >= abs_stft_data2

video_net = modifyresnet18()
audio_net = UNet7()
syn_net = synthesizer()

video_net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'video_net_params.pkl')))
audio_net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'audio_net_params.pkl')))
syn_net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'syn_net_params.pkl')))

print(mask)

result1 = np.zeros(stft_data.shape,dtype = complex)
result2 = np.zeros(stft_data.shape,dtype = complex)
result1[mask] = stft_data[mask]
result2[~mask] = stft_data[~mask]

'''
plt.subplot(1,3,1)
librosa.display.specshow((np.abs(stft_data)),y_axis='linear', x_axis='time')
plt.subplot(1,3,2)
librosa.display.specshow((np.abs(result1)),y_axis='linear', x_axis='time')
plt.subplot(1,3,3)
librosa.display.specshow((np.abs(result2)),y_axis='linear', x_axis='time')
plt.show()
'''

# result_data1 = librosa.istft(result1,hop_length = HOP_LENGTH,win_length=1022)
# result_data2 = librosa.istft(result2,hop_length = HOP_LENGTH,win_length=1022)

result_data1 = waveoperate.mask2wave(result1,'linear')
result_data2 = waveoperate.mask2wave(result2,'linear')

librosa.output.write_wav(os.path.join('./test','result1.wav'),result_data1,sr) 
librosa.output.write_wav(os.path.join('./test','result2.wav'),result_data2,sr) 



'''
for instrument in instrument_class:
	print(instrument)
	if os.path.exists(os.path.join(NEW_DIR,instrument)) == False:
		os.mkdir(os.path.join(NEW_DIR,instrument))
	files_dir = os.path.join(OLD_DIR,instrument)
	files = os.listdir(files_dir)
	for file in files:
		w,sr = librosa.load(os.path.join(files_dir,file),sr=SAMPLE_RATE)
		block_num = math.floor(len(w)/BLOCK_LENGTH)
		order = file[:-4]
		destnation = os.path.join(NEW_DIR,instrument,order)
		if os.path.exists(destnation) == False:
			os.mkdir(destnation)
		num = 0
		for i in range(block_num):
			num += 1
			# select the wave data
			data = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH-1]
			# STFT
			librosa.output.write_wav('raw.wav',data,sr) 
			stft_data = librosa.stft(data,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
			wave = librosa.istft(stft_data,hop_length = HOP_LENGTH)
			librosa.output.write_wav('new.wav',wave,sr) 
			wave = librosa.istft(np.abs(stft_data),hop_length = HOP_LENGTH)
			librosa.output.write_wav('abs.wav',wave,sr) 
			# log scale resample
			#print(sample_index)
			#stft_data = stft_data[sample_index,:]
			print(os.path.join(destnation,str(i)+'.wav'))
			exit()
			print(sample_index2)
			stft_data = stft_data[tuple(sample_index2),:]
			# save data
			np.save(os.path.join(destnation,str(i)),stft_data)
			
			# print the spectrum
			librosa.display.specshow((np.abs(stft_data)),y_axis='log', x_axis='time')
			print('done!')
			#print(np.shape(stft_data))
			plt.title('Power spectrogram')
			plt.colorbar(format='%+2.0f dB')
			plt.tight_layout()
			plt.show()

			# if you want to test the code, you can add this exit() and try one example 
			exit()

			# save the wave segment
			librosa.output.write_wav(os.path.join(destnation,str(i)+'.wav'),data,sr) 
'''