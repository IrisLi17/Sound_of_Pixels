# Sound_of_Pixels
Course project of Introduction to Visual and Audio System.

Pytorch implementation of [The Sound of Pixels](https://arxiv.org/abs/1804.03160).

### Environment
* python 3.5
* pytorch 0.4.1
* cuda 8.0
* cudnn 6.0
 
### Structure
* models/
* util/

### Analysis for STFT
* The wave data in dataset is sampled by a sample_rate of 44.1kHz
* According to the paper, we can divide the wave into segments around 6s. And then down-sample these segments to 11kHz. 
* Then we will get around 66k samples for each segment.
* With a 1202 long window, we can get 256 samples in time domain with a gap of 256 between each two windows.
* 1022+256*255=66302, which means we need to cut the wave at least each 6.027 seconds, we can take it as 6.05 seconds. 
* In the frequency domain, we first get 512 samples via DTFT, and then resample it with log f scale into 256 samples.