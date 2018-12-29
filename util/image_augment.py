import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def flip(imname, outname):
    image = cv2.imread(imname)
    flipped = cv2.flip(image, 1) # horizontally
    cv2.imwrite(outname, flipped)

def main(instrument_dir = '/data/liyunfei/dataset/video_3frames/trumpet'):
    cases = os.listdir(instrument_dir) # assume cases are numbered from 0
    case_num = len(cases)
    for case in cases:
        case_dir = os.path.join(instrument_dir, case)
        aug_case_dir = os.path.join(instrument_dir, str(int(case) + case_num))
        # print("case:",case,"new case:",str(int(case) + case_num))
        if not os.path.exists(aug_case_dir):
            os.mkdir(aug_case_dir)
        items = os.listdir(case_dir)
        for item in items:
            image = np.load(os.path.join(case_dir, item))
            # flipped_image = np.flip(image, axis=2)
            # flipped_image = np.rot90(image, axes=(1,2))
            # plt.imshow(flipped_image[0])
            # plt.show()
            # exit()
            aug_item_name = os.path.join(aug_case_dir, item)
            print("aug_item_name", aug_item_name)
            np.save(aug_item_name, image)

if __name__=='__main__':
    instrument_dir = r'D:\huyb\std\audio_spectrums_linear\saxophone'
    main(instrument_dir)