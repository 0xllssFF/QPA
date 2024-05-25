import os
import numpy as np
np.random.seed(666)
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
import random
import networkx as nx
import imp
import threading
from PIL import Image
import json
import math
from SDMs.BlackLight import BlackLight
from SDMs.PIHA import PIHA
from SDMs.QPA import QPA,GCN
from SDMs.QPABL import QPABL
from argparse import ArgumentParser
import time
import logging
import logging.handlers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



def main(args):


    config = json.load(open(args.config))
    eval_config = config["eval_config"]
    blacklight_config = config["blacklight_config"]
    piha_config = config["piha_config"]
    qpa_config = config["qpa_config"]

    num_instance = args.instance
    ttd = args.ttd
    topk = args.k
    attack_type = eval_config["attack"]
    attack_mode = eval_config["mode"]
    dataset = eval_config["dataset"]


    dataset_path = eval_config["dataset_path"]
    adv_query_path = "/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/{}/attack_query/{}/{}/".format(dataset, attack_type, attack_mode)
    selected_instance = random.sample(os.listdir(adv_query_path), num_instance)
    anomaly_rate = eval_config["anomaly_rate"]

    if not args.disable_logging:
        log_dir = os.path.join("/".join(args.config.split("/")[:-1]), 'logs/')
        filename=os.path.join(log_dir, f'log_{args.instance}.json')
        log = open(filename, 'a')
        # latency_name = os.path.join(log_dir, f'latency_{dataset}_woopt.json')
        # latency_log = open(latency_name, 'a')




    # Load dataset.
    if dataset == "mnist":
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])    
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.CenterCrop(35),
            transforms.ToTensor()])
    elif dataset == "gtsrb":
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset == "imagenet":
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    elif dataset == "iot_sqa":
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset == "celebahq":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError("Dataset not supported.")

    for num in selected_instance:
        print(num)
        query_path = os.path.join(adv_query_path, num)
        adv_query_filepath_list = sorted([os.path.join(query_path, f) for f in os.listdir(query_path) if f.endswith('.png')], key = lambda x: int(x.split('/')[-1].split('.')[0]))
        if len(adv_query_filepath_list) / anomaly_rate > args.max_queries:
            adv_query_filepath_list = adv_query_filepath_list[:int(args.max_queries * anomaly_rate)]
        adv_query = [utils.load_image(f, transform) for f in adv_query_filepath_list]
        total_query_num = math.ceil(len(adv_query) / anomaly_rate)
        normal_num = total_query_num - len(adv_query)
        print(normal_num, len(adv_query), total_query_num)
        if dataset == 'imagenet':
            selected_img = random.sample(range(0, 50000), normal_num)
            x, _ = utils.get_imagenet_list(selected_img, transform, dataset_path, mode = 'test')
        elif dataset == 'cifar10':
            selected_img = random.sample(range(2000, 50000), normal_num)
            # selected_img = random.sample(range(0, 50000), normal_num)
            x = utils.get_cifar10_list(selected_img, transform, dataset_path)
        elif dataset == "mnist":
            selected_img = random.sample(range(10000, 60000), normal_num)
            x = utils.get_mnist_list(selected_img, transform, dataset_path)
        elif dataset == "celebahq":
            selected_img = random.sample(range(5000, 30000), normal_num)
            # selected_img = random.sample(range(0, 30000), normal_num)
            x = utils.get_celebahq_list(selected_img, transform, dataset_path)
        inserted_position = sorted(random.sample(range(0, total_query_num), len(adv_query)))
        print(inserted_position)

        test_dataset = []
        for i in range(total_query_num):
            if i in inserted_position:
                test_dataset.append(adv_query.pop(0))
            else:
                test_dataset.append(x.pop(0))

        test_dataset = np.array(test_dataset)

        print(test_dataset.shape)
        
        attack_query = set(inserted_position)

        # blacklight
        print("start blacklight")
        tracker = BlackLight(blacklight_config)
        detected_query = set()
        start_time = time.time()
        for idx, query in tqdm(enumerate(test_dataset)):
            # start_time_per = time.time()
            similarity_result = tracker.resultsTopk(query, 1)
            if len(similarity_result) > 0:
                dist = similarity_result[0]
                if dist >= blacklight_config["threshold"]:
                    detected_query.add(idx)
                    # print("Image: {}, max match: {}".format(idx, dist))
            tracker.add(query)
            # end_time_per = time.time()
            # latency = end_time_per - start_time_per
            # latency_log_data = {'SDM': "BlackLight", 'Latency': latency, 'ID': idx}
            # json.dump(latency_log_data, latency_log)
            # latency_log.write('\n')
        end_time = time.time()

        
        if len(detected_query & attack_query) > 0:
            first_alert = int(np.where(np.array(inserted_position) == min(list(detected_query & attack_query)))[0][0])
            log_data = {'SDM': "Blacklight", 'Precision': len(detected_query & attack_query)/len(detected_query), 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': first_alert, 'number of query': total_query_num}
            json.dump(log_data, log)
            log.write('\n')
            # logging.info("BlackLight: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first alert: {}, number of query: {}".format(len(detected_query & attack_query)/len(detected_query), len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))
        else:
            log_data = {'SDM': "Blacklight", 'Precision': None , 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': None, 'number of query': total_query_num}
            json.dump(log_data, log)
            log.write('\n')

            logging.info("BlackLight: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first alert: {}, number of query: {}".format(None, len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))
        
        # # piha
        if dataset != "mnist":
            print("start PIHA")
            detected_query = set()
            tracker = PIHA(piha_config)
            start_time = time.time()
            for idx, query in tqdm(enumerate(test_dataset)):
                # start_time_per = time.time()
                similarity_result = tracker.resultsTopk(query, 1)
                if len(similarity_result) > 0:
                    dist = similarity_result[0]
                    if dist <= piha_config["threshold"]:
                        detected_query.add(idx)
                        # print("Image: {}, max match: {}".format(idx, dist))
                tracker.add(query)
                # end_time_per = time.time()
                # latency = end_time_per - start_time_per
                # latency_log_data = {'SDM': "PIHA", 'Latency': latency, 'ID': idx}
                # json.dump(latency_log_data, latency_log)
                # latency_log.write('\n')
            end_time = time.time()


            if len(detected_query & attack_query) > 0:
                # print(np.where(np.array(inserted_position) == min(list(detected_query))))
                first_alert = int(np.where(np.array(inserted_position) == min(list(detected_query & attack_query)))[0][0])
                log_data = {'SDM': "PIHA", 'Precision': len(detected_query & attack_query)/len(detected_query), 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': first_alert, 'number of query': total_query_num}
                json.dump(log_data, log)
                log.write('\n')
                # logging.info("PIHA: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first alert: {}, number of query: {}".format(len(detected_query & attack_query)/len(detected_query), len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))
            else:
                log_data = {'SDM': "PIHA", 'Precision': None , 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': None, 'number of query': total_query_num}
                json.dump(log_data, log)      
                log.write('\n')
                # logging.info("PIHA: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first alert: {}, number of query: {}".format(None, len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))

        # qpa
        if dataset != "mnist":
            print("start QPA")
            # prepare random sample data to initialize the cache
            train_dataset = []
            train_labels = []
            if dataset == 'imagenet':
                selected_img = random.sample(range(0, 50000), eval_config["init_num"])
                x, _ = utils.get_imagenet_list(selected_img, transform, dataset_path, mode = 'test')
            elif dataset == 'cifar10':
                selected_img = random.sample(range(0, 2000), eval_config["init_num"])
                x = utils.get_cifar10_list(selected_img, transform, dataset_path)
            elif dataset == "celebahq":
                selected_img = random.sample(range(2000, 5000), eval_config["init_num"])
                x = utils.get_celebahq_list(selected_img, transform, dataset_path)
            
            train_dataset.extend(x)
            train_dataset = np.array(train_dataset)

            tracker = QPA(train_dataset, qpa_config, ttd)
            match_list = []
            id = 0
            detected_query = set()
            start_time = time.time()
            for idx, query in tqdm(enumerate(test_dataset)):
                # start_time_per = time.time()
                match_num, match_idx, res = tracker.add(query)
                # if idx in inserted_position:
                #     print("({}, {}), {}".format(idx, match_idx,match_num))
                if res:
                    # print("Image: {}, match_idx: {}, max match: {}, attack_query: {}".format(idx, match_idx, match_num, res))
                    detected_query.add(idx)
                if idx % 500 == 0:
                    # LOGGER.info("PIHA detector started.")
                    tracker.detector(topk)
                    # LOGGER.info("PIHA detector ended.")
                # end_time_per = time.time()
                # latency = end_time_per - start_time_per
                # latency_log_data = {'SDM': "QPA", 'Latency': latency, 'ID': idx}
                # json.dump(latency_log_data, latency_log)
                # latency_log.write('\n')
            end_time = time.time()
            # print(min(detected_query))

        
            if len(detected_query & attack_query) > 0:
                first_alert = int(np.where(np.array(inserted_position) == min(list(detected_query & attack_query)))[0][0])
                for i in tracker.alerted_nodes:
                    if i - eval_config["init_num"] - 1 in attack_query:
                        detected_query.add(i - eval_config["init_num"] - 1)
                log_data = {'SDM': "QPA-woopt", 'Precision': len(detected_query & attack_query)/len(detected_query), 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': first_alert, 'number of query': total_query_num}
                json.dump(log_data, log)
                log.write('\n')
                
                # logging.info("QPA: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first aleat: {}, number of query: {}".format(len(detected_query & attack_query)/len(detected_query), len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))
            else:
                log_data = {'SDM': "QPA-woopt", 'Precision': None , 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': None, 'number of query': total_query_num}
                json.dump(log_data, log)              
                log.write('\n')

                # logging.info("QPA: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first aleat: {}, number of query: {}".format(None, len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))
        else:
            print("start BLQPA: {}".format(num))
            # prepare random sample data to initialize the cache
            train_dataset = []
            train_labels = []

            selected_img = random.sample(range(0, 10000), eval_config["init_num"])
            x = utils.get_mnist_list(selected_img, transform, dataset_path)
   
            train_dataset.extend(x)
            train_dataset = np.array(train_dataset)

            tracker = QPABL(train_dataset, qpa_config, ttd)
            match_list = []
            id = 0
            detected_query = set()
            start_time = time.time()
            for idx, query in enumerate(test_dataset):
                # start_time_per = time.time()
                match_num, match_idx, res = tracker.add(query)
                # if idx in inserted_position:
                    # print("({}, {}), {}".format(idx, match_idx, match_num))
                if res:
                    # print("Image: {}, match_idx: {}, max match: {}, attack_query: {}".format(idx, match_idx, match_num, res))
                    detected_query.add(idx)
                if idx % 500 == 0:
                    # LOGGER.info("PIHA detector started.")
                    tracker.detector(topk)
                    # LOGGER.info("PIHA detector ended.")
                # end_time_per = time.time()
                # latency = end_time_per - start_time_per
                # latency_log_data = {'SDM': "QPA", 'Latency': latency, 'ID': idx}
                # json.dump(latency_log_data, latency_log)
                # latency_log.write('\n')
            end_time = time.time()
            # print(min(detected_query))

        
            if len(detected_query & attack_query) > 0:
                first_alert = int(np.where(np.array(inserted_position) == min(list(detected_query & attack_query)))[0][0])
                for i in tracker.alerted_nodes:
                    if i - eval_config["init_num"] in attack_query:
                        detected_query.add(i - eval_config["init_num"])
                log_data = {'SDM': "QPA-woopt", 'Precision': len(detected_query & attack_query)/len(detected_query), 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': first_alert, 'number of query': total_query_num}
                json.dump(log_data, log)
                log.write('\n')
                # logging.info("QPA: Detect Precision: {}, Recall: {}, FPR: {}, time: {}, first aleat: {}, number of query: {}".format(len(detected_query & attack_query)/len(detected_query), len(detected_query & attack_query)/len(attack_query),len(detected_query - attack_query)/normal_num, end_time - start_time, first_alert, total_query_num))
            else:
                log_data = {'SDM': "QPA-woopt", 'Precision': None , 'Recall': len(detected_query & attack_query)/len(attack_query), 'FPR': len(detected_query - attack_query)/normal_num, 'time': end_time - start_time, 'first_alert': None, 'number of query': total_query_num}
                json.dump(log_data, log)                          
                log.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--instance', type=int)
    parser.add_argument('--max_queries', type=int)
    parser.add_argument('--disable_logging', action='store_true', default = False)
    parser.add_argument('--ttd', type = int, default = 10)
    parser.add_argument('--k', type = int, default = 10)


    main(parser.parse_args())