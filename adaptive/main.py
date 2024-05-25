import os
import warnings
import logging
import logging.handlers
import multiprocessing
import random
import json
import socket
from datetime import datetime
import time
from argparse import ArgumentParser
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from attacks.attacks import *
from models.statefuldefense import init_stateful_classifier
from utils import datasets
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import csv


warnings.filterwarnings("ignore")

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
        

class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        # probas = F.softmax(logis, dim=1)
        return logits

def main(args):
    # Set up logging and load config.
    if not args.disable_logging:
        random.seed(round(time.time() * 1000))
        log_dir = os.path.join("/".join(args.config.split("/")[:-1]), 'logs', args.config.split("/")[-1].split(".")[0])
        writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(
            filename=os.path.join(writer.log_dir, f'log_{args.start_idx}_{args.start_idx + args.num_images}.txt'),
            level=logging.INFO)
        logging.info(args)
        mylogger = open(os.path.join("/".join(args.config.split("/")[:-1]), 'logs', 'QPA.json'), 'a')



    config = json.load(open(args.config))
    model_config, attack_config = config["model_config"], config["attack_config"]
    distance_file = open(os.path.join("/".join(args.config.split("/")[:-1]), 'logs', 'Distance_{}.csv'.format(model_config['state']['type'])), 'a', newline = "")
    distance_writer = csv.writer(distance_file)
    distance_writer.writerow(['Distance'])

    sequence_distance_file = open(os.path.join("/".join(args.config.split("/")[:-1]), 'logs', 'Sequence_Distance_{}.csv'.format(model_config['state']['type'])), 'a', newline = "")
    sequence_distance_writer = csv.writer(sequence_distance_file)
    sequence_distance_writer.writerow(['Distance'])

    logging.info(model_config)
    logging.info(attack_config)

    device = 'cuda:0'
    # Load model.
    model = init_stateful_classifier(model_config, attack_config['attack'].lower(), attack_config['targeted'])
    model.eval()
    # model.to(device)

    # Load dataset.
    if model_config["dataset"] == "mnist":
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
    elif model_config["dataset"] == "cifar10":
        transform = transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.CenterCrop(35),
            transforms.ToTensor()])
    elif model_config["dataset"] == "gtsrb":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "imagenet":
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    elif model_config["dataset"] == "iot_sqa":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "celebahq":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError("Dataset not supported.")
    json_path = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/{}/{}.json'.format(model_config["dataset"], model_config["dataset"])
    images_json = list(json.load(open(json_path)).items())
    train_list = [os.path.join('/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/{}/{}'.format(model_config["dataset"], imgs[0])) for imgs in random.sample(images_json, 1000)]
    train_dataset = [transform(Image.open(image_path).convert("RGB")) for image_path in train_list]

    test_dataset = datasets.StatefulDefenseDataset(name=model_config["dataset"], transform=transform,
                                                   size=args.num_images, start_idx=args.start_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    if model_config["state"]["type"] == 'toolname':
        model.state_module.init_graph(model, train_dataset)
    
    if attack_config["attack"] == "natural_accuracy":
        natural_performance(model, test_loader)
    else:
        attack_loader(model, test_loader, model_config, attack_config, device, mylogger, distance_writer, sequence_distance_writer)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser()
    parser.add_argument('--disable_logging', action='store_true')
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--log_dir', type=str)

    main(parser.parse_args())
