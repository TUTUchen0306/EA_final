import numpy as np
from PIL import Image
from torchvision import transforms


def read_img(file_name, height=512, width=512):
    image = Image.open(file_name)

    image = image.convert("RGB")

    loader = transforms.Compose([transforms.Resize((height, width))])

    image = loader(image)

    image = np.asarray(image)
    image_array = []
    red_array, green_array, blue_array = [], [], []

    for h in range(height):
        tr, tg, tb = [], [], []
        for w in range(width):
            r, g, b = image[h][w]
            tr.append(r)
            tg.append(g)
            tb.append(b)

        red_array.append(tr)
        green_array.append(tg)
        blue_array.append(tb)

    image_array.append(red_array)
    image_array.append(green_array)
    image_array.append(blue_array)

    image_array = np.array([image_array])

    return image_array


def shape_to_512_512_3(pic):
    # shape to 512, 512, 3
    pic_reshape = np.zeros((512, 512, 3))
    pic_reshape[:, :, 0] = pic[0][0]
    pic_reshape[:, :, 1] = pic[0][1]
    pic_reshape[:, :, 2] = pic[0][2]
    return pic_reshape


def shape_to_1_3_512_512(pic):
    # shape to 1, 3, 512, 512
    pic_reshape = np.zeros((1, 3, 512, 512))
    pic_reshape[0, 0, :, :] = pic[:, :, 0]
    pic_reshape[0, 1, :, :] = pic[:, :, 1]
    pic_reshape[0, 2, :, :] = pic[:, :, 2]
    return pic_reshape
