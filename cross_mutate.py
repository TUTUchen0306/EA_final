from util import read_img
import random
import numpy as np
import numpy as np
import os
import cv2
from PIL import Image

# gaussian noise
def noisy(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = (image + gauss).astype(int)
    noisy = np.clip(noisy, 0, 255)
    return noisy


def noisy_img(file_name):
    pic = read_img(file_name)
    # shape to 512, 512, 3
    pic_reshape = np.zeros((512, 512, 3))
    pic_reshape[:, :, 0] = pic[0][0]
    pic_reshape[:, :, 1] = pic[0][1]
    pic_reshape[:, :, 2] = pic[0][2]
    # # show original image
    # Image.fromarray(pic_reshape.astype("uint8"), "RGB").show()
    newpic = noisy(pic_reshape)
    # shape to 1, 3, 512, 512
    newpic_reshape = np.zeros((1, 3, 512, 512))
    newpic_reshape[0, 0, :, :] = newpic[:, :, 0]
    newpic_reshape[0, 1, :, :] = newpic[:, :, 1]
    newpic_reshape[0, 2, :, :] = newpic[:, :, 2]
    # # show noisy image
    Image.fromarray(newpic.astype("uint8"), "RGB").show()
    return newpic_reshape


# 1 point crossover with probability p = 1.0, no mutation operator
def crossover(arr_left, arr_right):
    # change ndarray to 1d list
    arr_left = arr_left.ravel().tolist()
    arr_right = arr_right.ravel().tolist()
    # cut between cutPoint and cutPoint - 1
    cutPoint = random.randint(1, len(arr_left) - 1)
    print(cutPoint)
    lL, lR = arr_left[:cutPoint], arr_left[cutPoint:]
    rL, rR = arr_right[:cutPoint], arr_right[cutPoint:]
    lNew = lL + rR
    rNew = rL + lR
    print(len(lL), len(rR), len(lL + rR), len(rL), len(lR), len(rL + lR))
    lNew = np.asarray(lNew).reshape(1, 3, 512, 512)
    rNew = np.asarray(rNew).reshape(1, 3, 512, 512)
    return lNew, rNew


# generate random (1, 3, 512, 512) picture
pic1 = read_img("lighthouse.png")
pic2 = read_img("starry.jpg")
# shape (1, 3, 512, 512)
new_pic1 = noisy_img("lighthouse.png")
new_pic2 = noisy_img("starry.jpg")
