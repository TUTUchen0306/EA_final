import argparse
import numpy as np
import math
import torch
from torch import nn
from torchvision import models
from extract import FeatureExtractor
from util import read_img, show_img
from cross_mutate import noisy_img, crossover, mutation


parser = argparse.ArgumentParser(description="ES and GA picture style tranfer.")
parser.add_argument("--content", type=str, default="lighthouse.png", help="content img file name")
parser.add_argument("--style", type=str, default="starry.jpg", help="style img file name")
parser.add_argument("--alpha", type=float, default=2500.0, help="alpha value")
parser.add_argument("--beta", type=float, default=0.6, help="beta value")
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
    mse = nn.MSELoss(reduction='mean')
    c_l = torch.Tensor([0])
    c_l = c_l.to(device)
    for o, c, w in zip(origin, cur, weight):
        batch, channel, height, width = c.shape
        batch, channel, height = torch.Tensor([batch]), torch.Tensor([channel]), torch.Tensor([height])
        constant = torch.Tensor([1.0]) / (batch * channel * height)
        constant = constant.to(device)        
        err = mse(o, c).item()
        c_l += w * constant * err

    # print(f'content loss : {c_l[0]}')
    return c_l[0]


def gram_matrix(x):
    batch, channel, height, width = x.shape
    tx = x.view(channel, height * width)
    return torch.mm(tx, tx.t())


def style_loss(origin, cur):
    weight = [0.5, 1.0, 1.5, 3.0, 4.0]
    weight = torch.Tensor(weight)
    weight = weight.to(device)
    mse = nn.MSELoss(reduction='mean')
    t_l = torch.Tensor([0])
    t_l = t_l.to(device)
    for o, c, w in zip(origin, cur, weight):
        batch, channel, height, width = c.shape
        batch, channel, height = torch.Tensor([batch]), torch.Tensor([channel]), torch.Tensor([height])
        constant = torch.Tensor([1.0]) / (torch.Tensor([2.0]) * batch * channel * height)
        constant *= constant
        constant = constant.to(device)
        O = gram_matrix(o).to(device)
        C = gram_matrix(c).to(device)
        err = mse(O, C).item()
        t_l += math.sqrt(w * constant * err)

    # print(f'style loss : {t_l[0]}')

    return t_l[0]


def fitness(content, style, cur, extractor, times):
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

    tb = beta

    if times > 30:
        tb = math.sqrt(times) * 0.12

    return -(alpha * math.sqrt(times) * L_content + tb * L_style)

def train_GA(content_img, style_img, extractor, number_of_G, generation_times, point_place):
    content_generation, style_generation = [], []
    content = read_img(content_img)
    style = read_img(style_img)
    
    for _ in range(number_of_G):
        n1 = noisy_img(content_img)
        n2 = noisy_img(style_img)
        content_generation.append([n1, fitness(content, style, n1, extractor, 0)])
        style_generation.append([n2, fitness(content, style, n2, extractor, 0)])

    gen_size = len(content_generation)

    rec = []

    for times in range(generation_times):
        content_pop_pool, style_pop_pool = [], []
        for i in range(gen_size):
            choose = np.random.choice(gen_size-1, 2)
            if content_generation[choose[0]][1] > content_generation[choose[1]][1]:
                content_pop_pool.append(content_generation[choose[0]])
            else:
                content_pop_pool.append(content_generation[choose[1]])
            if style_generation[choose[0]][1] > style_generation[choose[1]][1]:
                style_pop_pool.append(style_generation[choose[0]])
            else:
                style_pop_pool.append(style_generation[choose[1]])           

        content_generation, style_generation = [], []
        rec.append([content, fitness(content, style, content, extractor, times)])

        if times > 30:
            lr = len(rec)
            c = np.random.choice(gen_size-1)
            rc = np.random.choice(lr-1)
            style_pop_pool[c] = rec[rc]

        for i in range(gen_size):
            choose = np.random.choice(gen_size-1, 2)
            c1, c2 = crossover(content_pop_pool[choose[0]][0], style_pop_pool[choose[1]][0], point_place, times)
            # c1 = mutation(c1)
            # c2 = mutation(c2)
            c1_fitness = fitness(content, style, c1, extractor, times)
            c2_fitness = fitness(content, style, c2, extractor, times)
            content_generation.append([c1, c1_fitness])
            style_generation.append([c2, c2_fitness])

        content_pop_pool = sorted(content_pop_pool, key=lambda x : x[1])

        style_pop_pool = sorted(style_pop_pool, key=lambda x: x[1])


        if(times % 10 == 0):
            print(f'times : {times}, largest content fitness value : {content_pop_pool[gen_size-1][1]}, largest style fitness value : {style_pop_pool[gen_size-1][1]}')
            show_img(content_pop_pool[gen_size-1][0],times)
            show_img(style_pop_pool[gen_size-1][0],times+1)

    # for i in range(len(generation)):
    #     show_img(generation[i][0])
    #     print(f'picture{i} : {generation[i][1]}')
        
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

    # content = read_img(content_img)
    # style = read_img(style_img)
    # cur = read_img("result.png")

    # print(fitness(content, style, cur, myextractor))

    train_GA(content_img, style_img, myextractor, 10, 1000, cut_point)
