# Pytorch DDP Analysis

## Introduction

This project studies how to distributed train model in `Pytorch` and analyzes its efficiency. We run the same code on a GPU, two GPUs and four GPUs (two GPUs per node), and record their runtime. 

Experiments show that training model on 2 GPUs can save 40 of time than that on single GPU, but when done on 4 GPUs (2 GPUs per node) it will cost more time. We think it is possible that our dataset (cifar100) is too small. Nonetheless, it suggests that we should run code on a single node instead of multiple nodes to achieve better performance.

In addition, We provide a template of `Pytorch` distributed training.

## Structure

```
Tamplate/
|-- data/
|
|-- models/
|   |-- __init__.py
|   |-- models.py
|
|-- scripts/
|   |-- train_single_gpu.sh  # script for training on single gpu
|   |-- train_single_node.sh  # script for training on single node with multi gpus
|   |-- train_multi_nodes.sh  # script for training on multi nodes
|
|-- dataset.py
|-- main.py
|-- metrics.py
|-- engine.py
|-- utils.py
|-- requirements.txt
|-- README
```

- `dataset.py`: loading data and doing data augmentatioin
- `models/models.py`: implement your model 
- `scripts`: scripts for starting training
    - `train_single_gpu.sh`: training model on single gpu
    - `train_single_node.sh`: trainging model on single node with multi gpus
    -  `train_multi_nodes.sh`: training on multi nodes. Execute this script on each node to start distrubuted training.

## Experiment on DDP

### Setting

- device: TITAN RTX
- model: resnet18
- dataset: cifar100
- epochs: 10
- batch size: 128
- dataset augmentation:
    ```python
    transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    ```

### Results

- single GPU: 555.0834 s
- 2 GPUs on single node (launched by `torch.distributed.launch`, will be deprecated) : 301.9986 s
- 2 GPUs on single node (launched by `torchrun`) : 324.2550 s
- 4 GPUs on two nodes (launched by `torchrun`) : 549.2544 s

## Usage

First, clone the repository locally:
```
git clone https://github.com/js2hou/Pytorch-DDP-Analysis.git
```
Then, install requirements:
```
pip install -r requirements.txt
```
Loading your dataset in `dataset.py` and implement your models in `models/models.py`.  Remember to modify the code for calling the model in `main.py`. Last, 

- run `./script/train_single_gpu.sh` for training on single gpu
- run `./script/train_single_node.sh` for training on multi gpus (recommended)
- run `./script/train_multi_nodes.sh` on each node for training on multi nodes


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.