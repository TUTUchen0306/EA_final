from util import read_img, shape_to_512_512_3, shape_to_1_3_512_512
import random
import numpy as np
import numpy as np
from PIL import Image
import copy
import math

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

def mutation(img):
    img_reshape = shape_to_512_512_3(img)
    new_img = noisy(img_reshape)
    new_img_reshape = shape_to_1_3_512_512(new_img)
    return new_img_reshape.astype(int)

# 1 point crossover with probability p = 1.0, no mutation operator
def crossover(arr_left, arr_right, argument, times):
    # three color in same pixel has different cutPoint
    if argument == "diff_point":
        # 3 color
        newL = np.zeros((1, 3, 512, 512))
        newR = np.zeros((1, 3, 512, 512))
        for color in range(len(arr_left[0]) / 128):
            # 512 row
            for row in range(len(arr_left[0][color])):
                for col in range(len(arr_left[0][color][row])):
                    l = "{0:08b}".format(int(arr_left[0][color][row][col].tolist()))
                    r = "{0:08b}".format(int(arr_right[0][color][row][col].tolist()))
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
        rg = random.randint(0, 5)
        group = pow(2, rg)
        # group = int(32 / math.sqrt(times+1))
        # if times == 30:
        #     group = 3
        # elif times == 50:
        #     group = 2
        # newL = copy.deepcopy(arr_left)
        # newR = copy.deepcopy(arr_right)
        # for color in range(len(arr_left[0])):
        # 512 row
        # crossover_number = int(512 * 512 / group)
        # for _ in range(crossover_number):
        #     nrow = random.randint(0, 512-1)
        #     ncol = random.randint(0, 512-1)
        #     cutPoint = random.randint(1, 8 - 1)
        #     for color in range(3):
        #         l = "{0:08b}".format(int(arr_left[0][color][nrow][ncol].tolist()))
        #         r = "{0:08b}".format(int(arr_right[0][color][nrow][ncol].tolist()))
        #         # cut between cutPoint and cutPoint - 1
        #         cutPoint = random.randint(1, len(l) - 1)
        #         lL, lR = l[:cutPoint], l[cutPoint:]
        #         rL, rR = r[:cutPoint], r[cutPoint:]
        #         lNew = lL + rR
        #         rNew = rL + lR
        #         newL[0][color][nrow][ncol] = int(lNew, 2)
        #         newR[0][color][nrow][ncol] = int(rNew, 2)

        newL = np.zeros((1, 3, 512, 512))
        newR = np.zeros((1, 3, 512, 512))


        for row in range(int(len(arr_left[0][0]) / group)):
            for col in range(int(len(arr_left[0][0][row]) / group)):
                # cut between cutPoint and cutPoint - 1
                cutPoint = random.uniform(-0.15, 1.15)

                # content_cross_row = np.random.choice(int(len(arr_left[0][0]) / group)-1)
                # content_cross_col = np.random.choice(int(len(arr_left[0][0][row]) / group)-1)

                # style_cross_row = np.random.choice(int(len(arr_left[0][0]) / group)-1)
                # style_cross_col = np.random.choice(int(len(arr_left[0][0][row]) / group)-1)

                for i in range(group):
                    for j in range(group):
                        nrow = row * group + i
                        ncol = col * group + j

                        # n_ccr = content_cross_row * group + i
                        # n_ccc = content_cross_col * group + j

                        # n_scr = style_cross_row * group + i
                        # n_scc = style_cross_col * group + j


                        for color in range(3):
                            # l = "{0:08b}".format(int(arr_left[0][color][nrow][ncol].tolist()))
                            # r = "{0:08b}".format(int(arr_right[0][color][nrow][ncol].tolist()))
                            # # cut between cutPoint and cutPoint - 1
                            # cutPoint = random.randint(1, len(l) - 1)
                            # lL, lR = l[:cutPoint], l[cutPoint:]
                            # rL, rR = r[:cutPoint], r[cutPoint:]
                            # lNew = lL + rR
                            # rNew = rL + lR
                            # newL[0][color][nrow][ncol] = int(lNew, 2)
                            # newR[0][color][nrow][ncol] = int(rNew, 2)

                            tl, tr = arr_left[0][color][nrow][ncol], arr_right[0][color][nrow][ncol]
                            # cc = arr_right[0][color][n_ccr][n_ccc]
                            # sc = arr_left[0][color][n_scr][n_scc]

                            nl = cutPoint * tl + (1 - cutPoint) * tr
                            nr = (1 - cutPoint) * tl + cutPoint * tr

                            if nl < 0 or nl > 255 or nr < 0 or nr > 255:
                                mx = [1, 1, 1, 0, 0, -1, -1, -1]
                                my = [1, 0, -1, 1, -1, 1, 0, -1]

                                l_mean, r_mean = 0, 0
                                pixel_num = 0

                                for mi in range(8):
                                    nx = nrow + mx[mi]
                                    ny = ncol + my[mi]

                                    if nx >= 512 or nx < 0 or ny >= 512 or ny < 0:
                                        continue
                                    l_mean += arr_left[0][color][nx][ny]
                                    r_mean += arr_right[0][color][nx][ny]
                                    pixel_num += 1

                                l_mean = int(l_mean / pixel_num)
                                r_mean = int(r_mean / pixel_num)

                                if nl < 0 or nl > 255:
                                    nl = l_mean
                                if nr < 0 or nr > 255:
                                    nr = r_mean

                            newL[0][color][nrow][ncol] = nl
                            newR[0][color][nrow][ncol] = nr

        return newL, newR


if __name__ == "__main__":
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
