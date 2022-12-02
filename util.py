import numpy as np
from PIL import Image
from torchvision import transforms

def read_img(file_name, height=512, weight=512):
	image = Image.open(file_name)

	image = image.convert('RGB')

	loader=transforms.Compose([transforms.Resize((height, weight))])
	
	image = loader(image)

	image = np.asarray(image)

	image_array = []
	red_array, green_array, blue_array = [], [], []

	for h in range(height):
		tr, tg, tb = [], [], []
		for w in range(weight):	
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