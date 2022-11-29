import argparse
import numpy as np
import torch
from torch import nn
from torchvision import models
from extract import FeatureExtractor

parser = argparse.ArgumentParser(description="ES and GA picture style tranfer.")
parser.add_argument('--alpha', type=float, default=8,
                    help="alpha value")
parser.add_argument('--beta', type=float, default=70,
                    help="beta value")
parser.add_argument('--algorithm', type=str, default="GA",
                    help="algorithm type")

alpha = 0
beta = 0
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("Using device: {}".format(device))

extract_list = ['0', '5', '10', '19', '28']


def content_loss(origin, cur):
    weight = [0.5, 1.0, 1.5, 3.0, 4.0]
    c_l = torch.Tensor([0])
    for o, c, w in zip(origin, cur, weight):
        c_l += w * torch.mean((o - c) ** 2)
    return c_l[0]

def gram_matrix(x):
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
    args = parser.parse_args()
    args_dict = vars(args)

    model = models.vgg16(pretrained=True)

    # change maxPool layers to AvgPool layers inorder to avoid fading

    for i, layer in model.features.named_children():
        if isinstance(layer, torch.nn.MaxPool2d):
            model.features[int(i)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    model = model.features
    model = model.to(device)

    alpha = args.alpha
    beta = args.beta

    myextractor = FeatureExtractor(model, extract_list)

