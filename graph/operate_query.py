import os
import sys
import numpy as np
import random 
dataset = 'imagenet'

data_path = "/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/attack_query/"

dir_list = os.listdir(data_path)
print(dir_list)

train_query = []
for dir in dir_list:
    if dir == 'boundary':
        is_target = 'untargeted'
    elif dir == 'hsja':
        is_target = 'targeted'
    elif dir == 'nesscore':
        is_target = 'targeted'
    elif dir == 'qeba':
        is_target = 'untargeted'
    elif dir == 'square':
        is_target = 'untargeted'
    elif dir == 'surfree':
        is_target = 'untargeted'
    instance_list = os.listdir(data_path + dir + '/' + is_target)
    print(dir, len(instance_list))
    total_queries = 0
    for instance in instance_list:
        instance_path = data_path + dir + '/' + is_target + '/' + instance
        query_list = os.listdir(instance_path)
        total_queries += len(query_list)
    print(total_queries / len(instance_list))

    train_instance = random.sample(instance_list, min(5, len(instance_list)))
    print(train_instance)
    train_query += [data_path + dir + '/' + is_target + '/' + instance for instance in train_instance]
    
for i, query in enumerate(train_query):
    print(query)
    os.system('cp -r ' + query + ' /home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/tmp/')
    os.system('mv ' + '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/tmp/' + query.split('/')[-1] + ' ' + '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/' + str(i))