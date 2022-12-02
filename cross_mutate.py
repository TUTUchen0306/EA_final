from util import read_img, shape_to_512_512_3, shape_to_1_3_512_512
import random
import numpy as np
import numpy as np
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
    pic_reshape = shape_to_512_512_3(pic)
    # # show original image
    # Image.fromarray(pic_reshape.astype("uint8"), "RGB").show()
    newpic = noisy(pic_reshape)
    # shape to 1, 3, 512, 512
    newpic_reshape = shape_to_1_3_512_512(newpic)
    # # show noisy image
    # Image.fromarray(newpic.astype("uint8"), "RGB").show()
    return newpic_reshape.astype(int)


# 1 point crossover with probability p = 1.0, no mutation operator
def crossover(arr_left, arr_right, argument):
    # three color in same pixel has different cutPoint
    if argument == "diff_point":
        # 3 color
        newL = np.zeros((1, 3, 512, 512))
        newR = np.zeros((1, 3, 512, 512))
        for color in range(len(arr_left[0])):
            # 512 row
            for row in range(len(arr_left[0][color])):
                for col in range(len(arr_left[0][color][row])):
                    l = "{0:08b}".format(arr_left[0][color][row][col].tolist())
                    r = "{0:08b}".format(arr_right[0][color][row][col].tolist())
                    # cut between cutPoint and cutPoint - 1
                    cutPoint = random.randint(1, len(l) - 1)
                    lL, lR = l[:cutPoint], l[cutPoint:]
                    rL, rR = r[:cutPoint], r[cutPoint:]
                    lNew = lL + rR
                    rNew = rL + lR
                    newL[0][color][row][col] = int(lNew, 2)
                    newR[0][color][row][col] = int(rNew, 2)
        return newL, newR
    # three color in same pixel has same cutPoint
    elif argument == "same_point":
        # 3 color
        newL = np.zeros((1, 3, 512, 512))
        newR = np.zeros((1, 3, 512, 512))
        # for color in range(len(arr_left[0])):
        # 512 row
        for row in range(len(arr_left[0][0])):
            for col in range(len(arr_left[0][0][row])):
                # cut between cutPoint and cutPoint - 1
                cutPoint = random.randint(1, 8 - 1)
                for color in range(3):
                    l = "{0:08b}".format(arr_left[0][color][row][col].tolist())
                    r = "{0:08b}".format(arr_right[0][color][row][col].tolist())
                    # cut between cutPoint and cutPoint - 1
                    cutPoint = random.randint(1, len(l) - 1)
                    lL, lR = l[:cutPoint], l[cutPoint:]
                    rL, rR = r[:cutPoint], r[cutPoint:]
                    lNew = lL + rR
                    rNew = rL + lR
                    newL[0][color][row][col] = int(lNew, 2)
                    newR[0][color][row][col] = int(rNew, 2)
        return newL, newR


# generate random (1, 3, 512, 512) picture
pic1 = read_img("lighthouse.png")
pic2 = read_img("starry.jpg")
# shape (1, 3, 512, 512)
new_pic1 = noisy_img("lighthouse.png")
new_pic2 = noisy_img("starry.jpg")

newL, newR = crossover(new_pic1, new_pic2, "diff_point")
# # uncomment to see examples
# newL_reshape = shape_to_512_512_3(newL)
# Image.fromarray(newL_reshape.astype("uint8"), "RGB").show()
# newR_reshape = shape_to_512_512_3(newR)
# Image.fromarray(newR_reshape.astype("uint8"), "RGB").show()
newL, newR = crossover(new_pic1, new_pic2, "same_point")
# newL_reshape = shape_to_512_512_3(newL)
# Image.fromarray(newL_reshape.astype("uint8"), "RGB").show()
# newR_reshape = shape_to_512_512_3(newR)
# Image.fromarray(newR_reshape.astype("uint8"), "RGB").show()
