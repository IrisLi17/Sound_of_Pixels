# this code should be run inside the dataset folder and up the audios folder
# the spectrum will be saved in NEW_DIR in npy format
# check the NEW_DIR and OLD_DIR to ensure you have put it in the write directory

import wave
import os
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np

NEW_DIR = './audio_spectrums'
OLD_DIR = './audios/solo' 
SAMPLE_RATE = 11000
BLOCK_LENGTH = 66302
WINDOW_SIZE = 1022
HOP_LENGTH = 256

frequencies = np.linspace(SAMPLE_RATE/2/512,SAMPLE_RATE/2,512)
log_freq = np.log10(frequencies)
sample_freq = np.linspace(log_freq[0],log_freq[-1],256)
sample_index = np.array([np.abs(log_freq-x).argmin() for x in sample_freq])
# prepare for log resample
sample_index2 =np.array(np.linspace(0,510,256)).astype(int)

if os.path.exists(NEW_DIR) == False:
	os.mkdir(NEW_DIR)

instrument_class = os.listdir(OLD_DIR)
for instrument in instrument_class:
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
		for i in range(block_num):
			# select the wave data
			data = w[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH]
			# STFT
			stft_data = librosa.stft(data,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
			# log scale resample
			stft_data = stft_data[sample_index,:]
			print(stft_data.shape)
			# save data
			np.save(os.path.join(destnation,str(i)),stft_data)
			
			# print the spectrum
			#librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_data),ref=np.max),y_axis='log', x_axis='time')
			#print('done!')
			#print(np.shape(stft_data))
			#plt.title('Power spectrogram')
			#plt.colorbar(format='%+2.0f dB')
			#plt.tight_layout()
			#plt.show()

			# if you want to test the code, you can add this exit() and try one example 
			exit()

			# save the wave segment
			#librosa.output.write_wav(os.path.join(destnation,str(i)+'.wav'),data,sr) 