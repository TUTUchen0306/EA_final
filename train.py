import torch
from torch import nn
from torchvision import models
from extract import FeatureExtractor

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("Using device: {}".format(device))

extract_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']


def content_loss(origin, cur):
	return

def style_loss(origin, cur):
    return

def fitness(content, style, cur, extractor):
    content_vec = extractor(content)
    style_vec = extractor(style)
    cur_vec = extractor(cur)

    L_content = content_loss(content_vec, cur_vec)
    L_style = style_loss(style_vec, cur_vec)

    return a * L_content + b * L_style

if __name__ == "__main__":

model = models.vgg16(pretrained=True)


# change maxPool layers to AvgPool layers inorder to avoid fading

for i, layer in model.features.named_children():
    if isinstance(layer, torch.nn.MaxPool2d):
        model.features[int(i)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

model = model.features
model = model.to(device)

myextractor = FeatureExtractor(model, extract_list)

