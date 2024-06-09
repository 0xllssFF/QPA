## Query Provenance Analysis: Efficient and Robust Defense against Query-based Black-box Attacks



### 1. Setup the Environment
```
$ conda env create -f environment.yml -n qpa
```

### 2. Usage of Code
#### 2.1 Prepare the dataset
We organize our dataset as [OARS](https://github.com/nmangaokar/ccs_23_oars_stateful_attacks/tree/main?tab=readme-ov-file#224-data). You have to set the path of your dataset in `utils/datasets.py`.


#### 2.2 Generate the attack query sequence
We generate the attack sequence without the SDMs. `no` indicates the `NoOpState` in `models/statefuldefense.py`. 
```
$ python main.py --config configs/{your_dataset}/no/{attack_type}/config.json --num_images 100 --start_idx 0
```
We store the sequence in `data/{yourdataset}/attack_query/`.

#### 2.3 Train the graph model
We randomly select the attack sequence for graph model training. We store the selected training sequence in `data/{yourdataset}/anomaly_query/`
```
$ cd graph/
### select the training data.
$ python operate_query.py 
### construct the query provenance graphs
$ python generate_graph.py 
### train the graph model
$ python graph_classifier.py 
```

#### 2.4 Non-adaptive attacks
```
$ cd non-adaptive/
$ python evaluation.py --config configs/{your_dataset}/{attack_type}/config.json --instance 100 --max_queries 50000 --ttd 15 --k 20
```
#### 2.5 Adaptive attacks
```
$ cd adaptive/
$ python main.py --config configs/{your_dataset}/toolname/{attack_type}/config.json --num_images 100 --start_idx 100 
```