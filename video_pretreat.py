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
import cv2

NEW_DIR = './video_3frames'
OLD_DIR = './videos/solo' 
BLOCK_TIME = 66302/11000
SAMPLE_RATE = 11000
FPS = 24
BLOCK_LENGTH = math.floor(BLOCK_TIME*FPS)
FRAME_INDEX = [0,math.floor((BLOCK_LENGTH-1)/2),BLOCK_LENGTH-1]

frequencies = np.linspace(SAMPLE_RATE/2/512,SAMPLE_RATE/2,512)
log_freq = np.log10(frequencies)
sample_freq = np.linspace(log_freq[0],log_freq[-1],256)
sample_index = [np.abs(log_freq-x).argmin() for x in sample_freq]
# prepare for log resample

if os.path.exists(NEW_DIR) == False:
	os.mkdir(NEW_DIR)

instrument_class = os.listdir(OLD_DIR)
for instrument in instrument_class:
	if os.path.exists(os.path.join(NEW_DIR,instrument)) == False:
		os.mkdir(os.path.join(NEW_DIR,instrument))
	files_dir = os.path.join(OLD_DIR,instrument)
	files = os.listdir(files_dir)
	for file in files:
		cap = cv2.VideoCapture(os.path.join(files_dir,file))
		frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		block_num = math.floor(frameCount/BLOCK_LENGTH)
		order = file[:-4]
		destnation = os.path.join(NEW_DIR,instrument,order)
		buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
		fc = 0
		ret = True
		while (fc < frameCount  and ret):
			ret, buf[fc] = cap.read()
			fc += 1
		cap.release()
		if os.path.exists(destnation) == False:
			os.mkdir(destnation)
		for i in range(block_num):
			temp = buf[i*BLOCK_LENGTH:(i+1)*BLOCK_LENGTH,:,:,:]
			result = temp[FRAME_INDEX,:,:,:]
			final = np.empty((len(FRAME_INDEX),224,112,3),np.dtype('uint8'))
			for p in range(0,len(FRAME_INDEX)):
				final[p,:,:,:]=cv2.resize(result[p,:,:,:],(112,224))
			np.save(os.path.join(destnation,str(i)),final)
			#exit()

			# save the wave segment
			#librosa.output.write_wav(os.path.join(destnation,str(i)+'.wav'),data,sr) 
