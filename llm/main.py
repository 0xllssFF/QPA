from datasets import load_dataset
import argparse
import pandas as pd
import os
import random
from PromptGraph import PromptGraph
import networkx as nx



def get_benign_prompt(dataset_name):
    if dataset_name == 'wildjailbreak':
        if not os.path.exists(f"benign_dataset/{dataset_name}"):
            os.makedirs(f"benign_dataset/{dataset_name}")
            # Load the WildJailbreak training set
            dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)

            # Load the WildJailbreak evaluation set
            # dataset = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)

            # Filter the dataset to include only rows where the label is 0
            filtered_dataset = dataset['train'].filter(lambda example: 'benign' in example['data_type'])

            filtered_dataset = pd.DataFrame(filtered_dataset)

            # Select the 'vanilla' and 'adversarial' fields and merge them into one column with the header 'prompt'
            filtered_dataset_vanilla = filtered_dataset[['vanilla']].rename(columns={'vanilla': 'prompt'})

            filtered_dataset_adversarial = filtered_dataset[['adversarial']].rename(columns={'adversarial': 'prompt'})

            # Concatenate the two datasets
            merged_dataset = pd.concat([filtered_dataset_vanilla, filtered_dataset_adversarial], ignore_index=True)
            merged_dataset = merged_dataset.drop_duplicates(subset=['prompt'])
            merged_dataset = merged_dataset[merged_dataset['prompt'].str.strip() != '']


            # Save the merged dataset
            merged_dataset.to_csv(f"benign_dataset/{dataset_name}/benign_prompt.csv", index=False)

        else:
            merged_dataset = pd.read_csv(f"benign_dataset/{dataset_name}/benign_prompt.csv")

        # Get the size of the filtered dataset
        merged_dataset_size = len(merged_dataset)

        print(f"Size of the filtered dataset: {merged_dataset_size}")



    return merged_dataset['prompt'].to_list()

def get_attack_prompt(attack_method, victim_model):
    file_path = f"attack_prompt/{attack_method}/{victim_model}/prompt.txt"
    prompt_list = []
    with open(file_path, 'r') as file:
        for line in file:
            attack_prompt = line.strip().replace('\\n','\n')
            # print(attack_prompt)
            prompt_list.append(attack_prompt)

    filtered_dataset_size = len(prompt_list)
    print(f"Size of the attack prompt: {filtered_dataset_size}")
    return prompt_list

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--benign_dataset", type=str, default="wildjailbreak")
    args.add_argument("--attack_method", type=str, default="PrivAgent")
    args.add_argument("--victim_model", type=str, default="Llama-3.1-8B-Instruct")

    args = args.parse_args()
    benign_dataset = args.benign_dataset
    attack_method = args.attack_method
    victim_model = args.victim_model

    # Get benign prompt dataset
    benign_prompt_dataset = get_benign_prompt(benign_dataset)

    # Get attack prompt 
    attack_prompt = get_attack_prompt(attack_method, victim_model)

    anomaly_rate = 0.01

    # Ensure the attack prompts are taken in order
    start_id = random.randint(0, len(attack_prompt) - 1 - int(len(benign_prompt_dataset) * anomaly_rate))
    attack_prompt = attack_prompt[start_id : start_id + int(len(benign_prompt_dataset) * anomaly_rate)]

    # Combine benign prompts and attack prompts with labels
    labeled_benign_prompts = [(prompt, 'benign') for prompt in benign_prompt_dataset]
    labeled_attack_prompts = [(prompt, 'attack') for prompt in attack_prompt]

    total_query_num = len(labeled_benign_prompts) + len(labeled_attack_prompts)

    inserted_position = sorted(random.sample(range(0, total_query_num), int(len(benign_prompt_dataset) * anomaly_rate)))

    combined_prompts_with_labels = []

    for i in range(total_query_num):
        if i in inserted_position:
            combined_prompts_with_labels.append(labeled_attack_prompts.pop(0))
        else:
            combined_prompts_with_labels.append(labeled_benign_prompts.pop(0))

    # Separate prompts and labels after shuffling
    combined_prompts, labels = zip(*combined_prompts_with_labels)


    print(f"total prompt:{len(combined_prompts)}")

    prompt_graph = PromptGraph()

    for prompt, label in combined_prompts_with_labels:
        prompt_graph.add_node(prompt, label)


    # Save the prompt graph to a dot file
    nx.drawing.nx_pydot.write_dot(prompt_graph.g, "prompt_graph.dot")

    anomaly_graphs = prompt_graph.detector()
    detected_prompt = []
    for x in anomaly_graphs:
        if len(x.graph.nodes()) > 10:
            detected_prompt.extend([int(i) - 1 for i in x.graph.nodes()])

    TP = set(inserted_position) & set(detected_prompt)
    FP = set(detected_prompt) - set(inserted_position)
    FN = set(inserted_position) - set(detected_prompt)

    print(f"Precision: {len(TP)/(len(TP)+len(FP))}")
    print(f"Recall: {len(TP)/(len(TP)+len(FN))}")
