import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt

global net
# global normalize
# global preprocess
global features_blobs
# global classes
global weight_softmax
labels_path = 'labels.json'
idxs = [401, 402, 486, 513, 558, 642, 776, 889]
names = ['accordion', 'acoustic_guitar', 'cello', 'trumpet', 'flute', 'xylophone', 'saxophone', 'violin']


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def hook_feature(module, input, output):
    global features_blobs
    features_blobs = output.data.cpu().numpy()


def load_model():
    global net
    # global normalize
    # global preprocess
    global features_blobs
    # global classes
    global weight_softmax
    # net = models.densenet161(pretrained=True)
    net = models.densenet161()
    model_dir = '../model'
    net.load_state_dict(torch.load(os.path.join(model_dir,'split_net_params.pkl')))
    ## save the model
    # torch.save(net.state_dict(), os.path.join(model_dir, 'split_net_params.pkl'))
    # print("model saved to " + str(model_dir) + '\n')
    ##
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    # classes = {int(key): value for (key, value)
    #            in json.load(open(labels_path, 'r')).items()}
    if torch.cuda.is_available():
        net = net.cuda()


# def get_CAM(imdir, savedir, imname):
#     img_pil = Image.open(os.path.join(imdir, imname))
#     img_tensor = preprocess(img_pil)
#     img_variable = Variable(img_tensor.unsqueeze(0))
#     if torch.cuda.is_available():
#         img_variable = img_variable.cuda()
#     img = cv2.imread(os.path.join(imdir, imname))
#     height, width, _ = img.shape
#     logit = net(img_variable)
#     h_x = F.softmax(logit, dim=1).data.squeeze()
#     if torch.cuda.is_available():
#         h_x = h_x.cpu()
#     probs1 = h_x.numpy()
#     probs = []
#     for i in range(0, 8):
#         print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
#
#         # CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])
#         # np.save(os.path.join(savedir, names[i], str(imname + 'CAMs.npy')), CAMs)
#         # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
#         # result = heatmap * 0.3 + img * 0.5
#         # cv2.imwrite(os.path.join(savedir, names[i], imname), result)
#
#         probs.append(probs1[idxs[i]])
#     probs = np.asarray(probs)
#     max_possible_idx = np.where(probs  == np.max(probs))[0] # interger between 0 and 7
#     CAM = returnCAM(features_blobs, weight_softmax, [idxs[max_possible_idx]])[0]
#     heat_point = np.where(CAM == np.max(CAM)) # (x,y)
#     return probs


# def main():
#     imdir = 'dataset/images/solo/acoustic_guitar/1'
#     load_model()
    # imlist = os.listdir(imdir)
    # probs = np.zeros([8])
    # for im in imlist:
    #     probs1 = get_CAM(imdir, 'results', im)
    #     probs = probs + np.array(probs1)
    # print(probs)
    # print(names)

def split_image(input_name):
    """
    :param input_name: duet image name
    :return: gap: optimal split x coordinate
    """
    load_model()
    image = Image.open(input_name)
    np_image = np.asarray(image)
    _width = np_image.shape[1]
    _left_half = np_image[:, :_width // 2, :]
    _right_half = np_image[:,_width // 2:, :]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    ## from numpy to Image
    _left_half = Image.fromarray(_left_half)
    _right_half = Image.fromarray(_right_half)
    _full = Image.fromarray(np_image)
    ## 
    left_image = preprocess(_left_half)
    right_image = preprocess(_right_half)
    full_image = preprocess(_full)
    # print('full image shape', full_image.shape)

    # plt.figure(1)
    # plt.subplot(1,3,1)
    # plt.imshow(np.transpose(full_image,(1,2,0)))
    # plt.subplot(1,3,2)
    # plt.imshow(np.transpose(left_image,(1,2,0)))
    # plt.subplot(1,3,3)
    # plt.imshow(np.transpose(right_image,(1,2,0)))
    # plt.show()
    left_img_variable = Variable(left_image.unsqueeze(0))
    right_img_variable = Variable(right_image.unsqueeze(0))
    full_img_variable = Variable(full_image.unsqueeze(0))
    if torch.cuda.is_available():
        left_img_variable = left_img_variable.cuda()
        right_img_variable = right_img_variable.cuda()
        full_img_variable = full_img_variable.cuda()
    left_logit = net(left_img_variable)
    right_logit = net(right_img_variable)
    full_logit = net(full_img_variable)
    left_h_x = F.softmax(left_logit, dim=1).data.squeeze()
    right_h_x = F.softmax(right_logit, dim=1).data.squeeze()
    full_h_x = F.softmax(full_logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        left_h_x = left_h_x.cpu()
        right_h_x = right_h_x.cpu()
        full_h_x = full_h_x.cpu()
    probs1 = left_h_x.numpy()
    probs2 = right_h_x.numpy()
    probs_full = full_h_x.numpy()
    left_probs = np.asarray([probs1[idxs[i]] for i in range(8)])
    right_probs = np.asarray([probs2[idxs[i]] for i in range(8)])
    full_probs = np.asarray([probs_full[idxs[i]] for i in range(8)])
    # print('full image probs', full_probs)
    # print('left image probs', left_probs)
    # print('right image probs', right_probs)
    left_max_possible_idx = np.where(left_probs == np.max(left_probs))[0]  # interger between 0 and 7
    right_max_possible_idx = np.where(right_probs == np.max(right_probs))[0]
    full_max_possible_idx = np.where(full_probs == np.max(full_probs))[0]
    # print('full max possible idx', full_max_possible_idx)
    left_CAM = cv2.resize(returnCAM(features_blobs, weight_softmax, [idxs[left_max_possible_idx[0]]])[0], left_image.shape[1:])
    right_CAM = cv2.resize(returnCAM(features_blobs, weight_softmax, [idxs[right_max_possible_idx[0]]])[0], right_image.shape[1:])
    full_CAM = cv2.resize(returnCAM(features_blobs, weight_softmax, [idxs[full_max_possible_idx[0]]])[0], full_image.shape[1:])
    # left_CAM =  returnCAM(features_blobs, weight_softmax, [idxs[left_max_possible_idx[0]]])[0]
    # right_CAM = returnCAM(features_blobs, weight_softmax, [idxs[right_max_possible_idx[0]]])[0]
    # full_CAM = returnCAM(features_blobs, weight_softmax, [idxs[full_max_possible_idx[0]]])[0]
    # plt.figure(1)
    # plt.subplot(1,3,1)
    # plt.imshow(full_CAM)
    # plt.subplot(1,3,2)
    # plt.imshow(left_CAM)
    # plt.subplot(1,3,3)
    # plt.imshow(right_CAM)
    # plt.colorbar()
    plt.show()
    fraction = 0.8
    left_heat_point = np.where(left_CAM >= np.max(left_CAM) * fraction)  # ((y1,y2,...),(x1,x2,...))
    left_heat_x = np.mean(left_heat_point[1])
    right_heat_point = np.where(right_CAM >= np.max(left_CAM) * fraction)
    right_heat_x = np.mean(right_heat_point[1])
    gap = math.floor(_width / 4 + (left_heat_x + right_heat_x) / 2)
    return gap


if __name__ == '__main__':
    input_image = r'D:\huyb\std\testset25\testimage\acoustic_guitar_2_violin_1\000128.jpg'
    split_image(input_image)
