import numpy as np
from PIL import Image
from torchvision import transforms


def read_img(file_name):
    image = Image.open(file_name)

    image = image.convert("RGB")

    loader = transforms.Compose([transforms.Resize((512, 512))])

    image = loader(image)

    image = np.asarray(image)
   
    image_array = shape_to_1_3_512_512(image)

    return image_array

def show_img(img, times, name):
    img = shape_to_512_512_3(img)
    Image.fromarray(img.astype("uint8"), "RGB").save(f'{name}_{times}.jpg')

    return


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
