import argparse
import numpy as np
import math
import torch
from torch import nn
from torchvision import models
from extract import FeatureExtractor
from util import read_img, show_img
from cross_mutate import noisy_img, crossover


parser = argparse.ArgumentParser(description="ES and GA picture style tranfer.")
parser.add_argument("--content", type=str, default="lighthouse.png", help="content img file name")
parser.add_argument("--style", type=str, default="starry.jpg", help="style img file name")
parser.add_argument("--alpha", type=float, default=1.0, help="alpha value")
parser.add_argument("--beta", type=float, default=1.0, help="beta value")
parser.add_argument("--algorithm", type=str, default="GA", help="algorithm type")
parser.add_argument("--cut_point", type=str, default="same_point", help="cut point place")


alpha, beta = 0, 0
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Using device: {}".format(device))

extract_list = ["0", "5", "10", "19", "28"]


def content_loss(origin, cur):
    weight = [0.5, 1.0, 1.5, 3.0, 4.0]
    weight = torch.Tensor(weight)
    weight = weight.to(device)
    mse = nn.MSELoss()
    c_l = torch.Tensor([0])
    c_l = c_l.to(device)
    for o, c, w in zip(origin, cur, weight):
        batch, channel, height, width = c.shape
        batch, channel, height = torch.Tensor([batch]), torch.Tensor([channel]), torch.Tensor([height])
        constant = torch.Tensor([1.0]) / (batch * channel * height)
        constant = constant.to(device)        
        err = mse(o, c).item()
        c_l += w * constant * err
    return c_l[0]


def gram_matrix(x):
    batch, channel, height, width = x.shape
    tx = x.view(channel, height * width)
    return torch.mm(tx, tx.t())


def style_loss(origin, cur):
    weight = [0.5, 1.0, 1.5, 3.0, 4.0]
    weight = torch.Tensor(weight)
    weight = weight.to(device)
    mse = nn.MSELoss()
    t_l = torch.Tensor([0])
    t_l = t_l.to(device)
    for o, c, w in zip(origin, cur, weight):
        batch, channel, height, width = c.shape
        batch, channel, height = torch.Tensor([batch]), torch.Tensor([channel]), torch.Tensor([height])
        constant = torch.Tensor([1.0]) / (batch * channel * height)
        constant = constant.to(device)
        O = gram_matrix(o).to(device)
        C = gram_matrix(c).to(device)
        err = mse(O, C).item()
        t_l += w * constant * err
    return t_l[0]


def fitness(content, style, cur, extractor):
    content = torch.Tensor(content)
    style = torch.Tensor(style)
    cur = torch.Tensor(cur)

    content = content.to(device)
    style = style.to(device)
    cur = cur.to(device)

    content_vec = extractor(content)
    style_vec = extractor(style)
    cur_vec = extractor(cur)

    L_content = content_loss(content_vec, cur_vec)
    L_style = style_loss(style_vec, cur_vec)

    return -(alpha * L_content + beta * L_style)

def train_GA(content_img, style_img, extractor, number_of_G, generation_times, point_place):
    generation = []

    content = read_img(content_img)
    style = read_img(style_img)
    
    for _ in range(number_of_G):
        n1 = noisy_img(content_img)
        n2 = noisy_img(style_img)
        generation.append([n1, fitness(content, style, n1, extractor)])
        generation.append([n2, fitness(content, style, n2, extractor)])

    gen_size = len(generation)

    for times in range(generation_times):
        pop_pool = []
        for i in range(gen_size):
            pop_pool.append(generation[i])
        generation = []
        for i in range(int(gen_size/2)):
            choose = np.random.choice(gen_size-1, 2)
            c1, c2 = crossover(pop_pool[choose[0]][0], pop_pool[choose[1]][0], point_place)
            c1_fitness = fitness(content, style, c1, extractor)
            c2_fitness = fitness(content, style, c2, extractor)
            generation.append([c1, c1_fitness])
            generation.append([c2, c2_fitness])

        pop_pool = sorted(pop_pool, key=lambda x: x[1])
        print(f'times : {times}, largest fitness value : {pop_pool[gen_size-1][1]}')


    for i in range(len(generation)):
        show_img(generation[i][0])
        print(f'picture{i} : {generation[i][1]}')
        
    return



if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parser.parse_args()
    args_dict = vars(args)

    model = models.vgg16(pretrained=True)

    # change maxPool layers to AvgPool layers inorder to avoid fading

    for i, layer in model.features.named_children():
        if isinstance(layer, torch.nn.MaxPool2d):
            model.features[int(i)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    model = model.features
    model = model.to(device)

    alpha, beta = args.alpha, args.beta
    content_img, style_img = args.content, args.style
    cut_point = args.cut_point

    myextractor = FeatureExtractor(model, extract_list)

    train_GA(content_img, style_img, myextractor, 10, 10, cut_point)
