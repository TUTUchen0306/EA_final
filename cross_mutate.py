from util import read_img
import random
import numpy as np


def init_gen():
    arr = []
    for i in range(512 * 512 * 3):
        tmp = random.randint(0, 255)
        arr.append(tmp)
    arr = np.asarray(arr)
    arr = arr.reshape(1, 3, 512, 512)
    return arr


# 1 point crossover with probability p = 1.0, no mutation operator
def crossover(arr_left, arr_right):
    # change ndarray to 1d list
    arr_left = arr_left.ravel().tolist()
    arr_right = arr_right.ravel().tolist()
    # cut between cutPoint and cutPoint - 1
    cutPoint = random.randint(1, len(arr_left) - 1)
    # print(cutPoint)
    lL, lR = arr_left[:cutPoint], arr_left[cutPoint:]
    rL, rR = arr_right[:cutPoint], arr_right[cutPoint:]
    lNew = lL + rR
    rNew = rL + lR
    # print(len(lL), len(rR), len(lL + rR), len(rL), len(lR), len(rL + lR))
    lNew = np.asarray(lNew).reshape(1, 3, 512, 512)
    rNew = np.asarray(rNew).reshape(1, 3, 512, 512)
    return lNew, rNew


# generate random (1, 3, 512, 512) picture
pic = init_gen()
pic1, pic2 = crossover(pic, pic)
# print(pic)
print(pic1.shape)
