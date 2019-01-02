import torch
import os 
import sys
import numpy as np
sys.path.append(r'D:\huyb\std\Sound_of_Pixels')
from models.models import modifyresnet18, UNet7, synthesizer
from util.datahelper import image_normalization
from librosa import amplitude_to_db
import matplotlib.pyplot as plt


def mix_spect_input(spect_input):
    # temp = np.sum(spect_input, axis=0)
    # return temp[np.newaxis, :]
    return amplitude_to_db(np.absolute(np.sum(spect_input, axis=0)), ref=np.max)


names = ['accordion', 'acoustic_guitar', 'cello', 'trumpet', 'flute', 'xylophone', 'saxophone', 'violin']
features = np.zeros((len(names),16))
audio_map = np.zeros((len(names)-1,16,256,256))
final_masks = np.zeros((len(names),256,256))
batch_size = 1
model_dir = r'D:\huyb\std\model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_net = modifyresnet18(batch_size).to(device)
audio_net = UNet7().to(device)
syn_net = synthesizer().to(device)
if os.path.exists(os.path.join(model_dir, '18video_net_params.pkl')) and os.path.exists(
        os.path.join(model_dir, '18audio_net_params.pkl')) and os.path.exists(
    os.path.join(model_dir, '18syn_net_params.pkl')):
    print('load params!')
    video_net.load_state_dict(torch.load(os.path.join(model_dir, '18video_net_params.pkl')))
    audio_net.load_state_dict(torch.load(os.path.join(model_dir, '18audio_net_params.pkl')))
    syn_net.load_state_dict(torch.load(os.path.join(model_dir, '18syn_net_params.pkl')))
    video_net.eval()
    audio_net.eval()
    syn_net.eval()

for idx in range(len(names)-1):
    for iteration in range(1,41):
        video_input1 = np.load('D:\\huyb\\std\\video_3frames\\'+names[idx]+'\\'+str(iteration)+'\\1.npy')
        video_input2 = np.load('D:\\huyb\\std\\video_3frames\\'+names[idx+1]+'\\'+str(iteration)+'\\2.npy')
        video_input1 = np.transpose(video_input1,(0,3,1,2))
        video_input2 = np.transpose(video_input2,(0,3,1,2))
        audio_input1 = np.load('D:\\huyb\\std\\audio_spectrums_linear\\'+names[idx]+'\\'+str(iteration)+'\\1.npy')
        audio_input2 = np.load('D:\\huyb\\std\\audio_spectrums_linear\\'+names[idx+1]+'\\'+str(iteration)+'\\2.npy')
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

        synspect_input = mix_spect_input(spect_input_mini)
        print(image_input_mini.shape)
        exit()
        image_input = image_normalization(image_input_mini).to(device)
        synspect_input = torch.from_numpy(synspect_input).to(device)

        out_audio_net = audio_net.forward(synspect_input)
        if iteration == 1:
            audio_map[idx,:,:,:] += out_audio_net[0,:,:,:].detach().cpu().numpy()

        for i in range(2):
            out_video_net = video_net.forward(image_input[i, :, :, :, :])
            temp = out_video_net * out_audio_net
            temp = torch.transpose(temp, 1, 2)
            temp = torch.transpose(temp, 2, 3)
            syn_act = syn_net.forward(temp)  # sigmoid logits
            syn_act = torch.transpose(syn_act, 2, 3)
            syn_act = torch.transpose(syn_act, 1, 2)
            if i==0 or (i==1 and idx==len(names)-2):
                features[idx+i,:] += out_video_net[0,:,0,0].detach().cpu().numpy()
                if iteration == 1:
                    final_masks[idx+i,:,:] += syn_act[0,0,:,:].detach().cpu().numpy()          
            
    # audio_map[idx,:,:,:] /= 10
    features[idx,:] /= 20
print("finish compute")
for i in range(len(names)):
    plt.figure()
    plt.bar(range(16),features[i,:])
    plt.title('feature of '+names[i])
    plt.savefig(names[i]+'.jpg')
    plt.close()
print("save bars")
for i in range(len(names)-1):
    plt.figure()
    plt.title('audionet output for '+names[i]+'_'+names[i+1])
    for x in range(4):
        for y in range(4):
            plt.subplot(4,4,4*x+y+1)
            plt.imshow(audio_map[i,4*x+y,:,:])   
    plt.savefig(names[i]+'_'+names[i+1]+'.jpg')
    plt.close()
print("save audio outputs")
for i in range(len(names)):
    plt.figure()
    plt.imshow(final_masks[i,:,:])
    plt.title('mask of '+names[i])
    plt.savefig('mask_'+names[i]+'.jpg')
    plt.close()
print("save masks")
