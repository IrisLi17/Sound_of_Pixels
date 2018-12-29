import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_image(input_name):
    image = cv2.imread(input_name)
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges = np.absolute(edges).astype(np.uint8)
    plt.subplot(1,3,1)
    plt.imshow(gray)
    plt.subplot(1,3,2)
    plt.imshow(edges)
    sums = np.sum(edges, axis=0)
    gap = np.where(sums == np.max(sums))[0]
    return gap

if __name__ == '__main__':
    input_image = r'D:\huyb\std\testset25\testimage\accordion_1_saxophone_1\000001.jpg'
    split_image(input_image)

