import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

# global net
# global normalize
# global preprocess
global features_blobs
# global classes
# global weight_softmax
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
    # global net
    # global normalize
    # global preprocess
    global features_blobs
    # global classes
    # global weight_softmax
    net = models.densenet161(pretrained=True)
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
    classes = {int(key): value for (key, value)
               in json.load(open(labels_path, 'r')).items()}
    if torch.cuda.is_available():
        net = net.cuda()


def get_CAM(imdir, savedir, imname):
    img_pil = Image.open(os.path.join(imdir, imname))
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable = img_variable.cuda()
    img = cv2.imread(os.path.join(imdir, imname))
    height, width, _ = img.shape
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x = h_x.cpu()
    probs1 = h_x.numpy()
    probs = []
    for i in range(0, 8):
        print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))

        # CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])
        # np.save(os.path.join(savedir, names[i], str(imname + 'CAMs.npy')), CAMs)
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # cv2.imwrite(os.path.join(savedir, names[i], imname), result)

        probs.append(probs1[idxs[i]])
    probs = np.asarray(probs)
    max_possible_idx = np.where(probs  == np.max(probs)) # interger between 0 and 7
    CAM = returnCAM(features_blobs, weight_softmax, [idxs[max_possible_idx]])[0]
    heat_point = np.where(CAM == np.max(CAM)) # (x,y)
    return probs


def main():
    imdir = 'dataset/images/solo/acoustic_guitar/1'
    load_model()
    imlist = os.listdir(imdir)
    probs = np.zeros([8])
    for im in imlist:
        probs1 = get_CAM(imdir, 'results', im)
        probs = probs + np.array(probs1)
    print(probs)
    print(names)

def split_image(net, input_name):
    """
    :param net: pretrained image classification model
    :param input_name: duet image name
    :return: optimal split x coordinate
    """
    image = Image.open(input_name)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    image = preprocess(image)
    img_variable = Variable(image.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable = img_variable.cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x = h_x.cpu()
    probs1 = h_x.numpy()
    probs = np.asarray([probs1[idxs[i]] for i in range(8)])
    max_possible_idx = np.where(probs == np.max(probs))  # interger between 0 and 7
    CAM = returnCAM(features_blobs, weight_softmax, [idxs[max_possible_idx]])[0]
    heat_point = np.where(CAM == np.max(CAM))  # (x,y)

if __name__ == '__main__':
    main()
