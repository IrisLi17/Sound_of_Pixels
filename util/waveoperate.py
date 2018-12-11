<<<<<<< HEAD
import numpy as np
import librosa


def mask2wave(mask_stft, WINDOW_SIZE=1022, HOP_LENGTH=256, FREQ_LENGTH=512, SAMPLE_RATE=11000,type):
    frequencies = np.linspace(SAMPLE_RATE / 2 / FREQ_LENGTH, SAMPLE_RATE / 2, FREQ_LENGTH)
    log_freq = np.log10(frequencies)
    sample_freq = np.linspace(log_freq[0], log_freq[-1], 256)
    sample_index = np.array([np.abs(log_freq - x).argmin() for x in sample_freq])
    sample_index2 = np.array(np.linspace(0,510,256)).astype(int)

    result_stft = np.zeros([FREQ_LENGTH, HOP_LENGTH], dtype=complex)
    # print(result_stft.shape)
    if type == 'log':
        result_stft[sample_index, :] = mask_stft
    elif type == 'linear':
        result_stft[sample_index2,:] = mask_stft

    has_value = np.zeros([FREQ_LENGTH], dtype=int)
    has_value[sample_index] = 1

    result_stft = upsample(result_stft, has_value)
    result_wave = librosa.istft(result_stft, hop_length=HOP_LENGTH, center=False)
    # librosa.output.write_wav('1.wav',result_wave,sr=SAMPLE_RATE)
    return result_wave


def upsample(data, binary_index, method='neighbor_average'):
    if method == 'neighbor_average':
        last_zero = -2
        for idx in range(0, len(binary_index)):
            if binary_index[idx] == 0:
                if idx == last_zero + 1:
                    data1 = data[idx - 1, :]
                    data[idx, :] = (data1 + data2) / 2
                else:
                    last_zero = idx
                    data1 = data[idx - 1, :]
                    for j in range(idx + 1, len(binary_index)):
                        if binary_index[j] == 1:
                            data2 = data[j, :]
                            break
                    data[idx, :] = (data1 + data2) / 2
                    binary_index[idx] = 1
        return data
=======
import numpy as np
import librosa


def mask2wave(mask_stft, type, WINDOW_SIZE=1022, HOP_LENGTH=256, FREQ_LENGTH=512, SAMPLE_RATE=11000):
    frequencies = np.linspace(SAMPLE_RATE / 2 / FREQ_LENGTH, SAMPLE_RATE / 2, FREQ_LENGTH)
    log_freq = np.log10(frequencies)
    sample_freq = np.linspace(log_freq[0], log_freq[-1], 256)
    sample_index = np.array([np.abs(log_freq - x).argmin() for x in sample_freq])
    sample_index2 = np.array(np.linspace(0,510,256)).astype(int)

    result_stft = np.zeros([FREQ_LENGTH, HOP_LENGTH], dtype=complex)
    # print(result_stft.shape)
    if type == 'log':
        result_stft[sample_index, :] = mask_stft
    elif type == 'linear':
        result_stft[sample_index2,:] = mask_stft

    has_value = np.zeros([FREQ_LENGTH], dtype=int)
    has_value[sample_index] = 1

    result_stft = upsample(result_stft, has_value)
    result_wave = librosa.istft(result_stft, hop_length=HOP_LENGTH, center=False)
    # librosa.output.write_wav('1.wav',result_wave,sr=SAMPLE_RATE)
    return result_wave


def upsample(data, binary_index, method='neighbor_average'):
    if method == 'neighbor_average':
        last_zero = -2
        for idx in range(0, len(binary_index)):
            if binary_index[idx] == 0:
                if idx == last_zero + 1:
                    data1 = data[idx - 1, :]
                    data[idx, :] = (data1 + data2) / 2
                else:
                    last_zero = idx
                    data1 = data[idx - 1, :]
                    for j in range(idx + 1, len(binary_index)):
                        if binary_index[j] == 1:
                            data2 = data[j, :]
                            break
                    data[idx, :] = (data1 + data2) / 2
                    binary_index[idx] = 1
        return data
>>>>>>> af4db0e244c43189c00ee9c88e0ac348ace1ffe2
