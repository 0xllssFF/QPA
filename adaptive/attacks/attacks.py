import logging
import time
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score

from seed import seed_everything
from attacks.Attack import AttackError

from attacks.adaptive.Square import Square
from attacks.adaptive.NESScore import NESScore
from attacks.adaptive.HSJA import HSJA
from attacks.adaptive.QEBA import QEBA
from attacks.adaptive.SurFree import SurFree
from attacks.adaptive.Boundary import Boundary
import json

@torch.no_grad()
def natural_performance(model, loader):
    logging.info("Computing natural accuracy")
    y_true, y_pred = [], []
    pbar = tqdm(range(0, len(loader)), colour="red")
    for i, (x, y, p) in (enumerate(loader)):
        start = time.time()
        logits, is_cache = model(x)
        end = time.time()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()

        logging.info(
            f"True Label : {y[0]} | Predicted Label : {preds[0]} | is_cache : {is_cache[0]} | latency : {end - start}")

        if model.config["action"] == "rejection":
            preds = [preds[j] if not is_cache[j] else -1 for j in range(len(preds))]
        true = y.detach().cpu().numpy().tolist()
        y_true.extend(true)
        y_pred.extend(preds)
        pbar.update(1)
        pbar.set_description(
            "Running accuracy: {} | hits : {}".format(accuracy_score(y_true, y_pred), model.cache_hits))
    logging.info("FINISHED")
    return accuracy_score(y_true, y_pred)


# @torch.no_grad()
def attack_loader(model, loader, model_config, attack_config, device, mylogger, distance_writer, sequence_distance_writer):
    # Load attack
    try:
        attacker = globals()[attack_config['attack']](model, model_config, attack_config)
    except KeyError:
        raise NotImplementedError(f'Attack {attack_config["attack"]} not implemented.')

    if attack_config['targeted']:
        target_labels = []
        for _, (_, y, p) in enumerate(loader):
            target_label = y.item()
            while target_label == y.item():
                target_label = np.random.randint(0, len(loader.dataset.targeted_dict))
            target_labels.append(target_label)
    else:
        target_labels = None

    # Run attack and compute adversarial accuracy
    y_true, y_pred = [], []
    pbar = tqdm(loader, colour="yellow")
    for i, (x, y, p) in enumerate(pbar):
        x = x
        y = y
        seed_everything()
        try:
            y_t = model.model(x).argmax(dim=1)
            if y_t != y:
                x_adv = x
            elif attack_config['targeted']:
                y_target = target_labels[i]
                x_adv_init = loader.dataset.initialize_targeted(y_target)
                y_target = torch.tensor([y_target])
                x_adv = attacker.attack_targeted(x, y_target, x_adv_init)
            else:
                x_adv = attacker.attack_untargeted(x, y)
        except AttackError as e:
            print(e)
            x_adv = x

        # x_adv = x_adv.to(device)
        # if attacker.get_cache_hits() == 0:
        #     continue
        x_adv = x_adv
        logits = model.model(x_adv)
        # preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        preds = torch.argmax(logits, dim=1).numpy().tolist()
        # true = y.detach().cpu().numpy().tolist()
        true = y.numpy().tolist()

        y_true.extend(true)
        y_pred.extend(preds)
        if attack_config['targeted']:
            print("Y_true: {}, Y_pred: {}, Y_target: {}".format(y_true, y_pred, y_target))
        else:
            print("Y_true: {}, Y_pred: {}".format(y_true, y_pred))
        pbar.set_description("Running Accuracy: {} ".format(accuracy_score(y_true, y_pred)))
        logging.info(
            f"True Label : {true[0]} | Predicted Label : {preds[0]} | Cache Hits / Total Queries : {attacker.get_cache_hits()} / {attacker.get_total_queries()}")
        logs = {"True Label":true[0], "Predicted Label": preds[0], "Cache Hit": attacker.get_cache_hits(), "Total Query": attacker.get_total_queries(), "Coverage": attacker.get_cache_hits() / attacker.get_total_queries()}
        json.dump(logs, mylogger)  
        mylogger.write("\n")
        # for i in model.distances:
        #     distance_writer.writerow([i])
        # for i in model.sequence_metric:
        #     sequence_distance_writer.writerow([i])
        attacker.reset()
    logging.info("FINISHED")
