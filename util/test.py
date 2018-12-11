import numpy as np
import waveoperate

a = np.load('0.npy')
mask = np.ones(a.shape,dtype = int)
print(a.shape,mask.shape)
c,d = waveoperate.mask2wave(a,mask)
print(c[90:110,1])
result = waveoperate.upsample(c,d)
print(result[90:110,1])