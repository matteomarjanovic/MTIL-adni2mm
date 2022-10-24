import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

from utils import read_json
from model_wrapper import CNN_Wrapper
import torch
import torch.multiprocessing as multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import time
torch.backends.benchmark = True

import sys
import wandb
import argparse
from datetime import datetime
import random
import numpy as np


def cnn_main(config, process, gpu_index, fold_index):
    cnn_setting = config['cnn']

    # CNN_Wrapper is used to wrap the model and its training and testing. The function used for cross-validation is
    # CNN_Wrapper.cross_validation.

    # with torch.cuda.device(gpu_index):
    cnn = CNN_Wrapper(fil_num=cnn_setting['fil_num'],
                        drop_rate=cnn_setting['drop_rate'],
                        batch_size=cnn_setting['batch_size'],
                        balanced=cnn_setting['balanced'],
                        learn_rate=cnn_setting['learning_rate'],
                        train_epoch=cnn_setting['train_epochs'],
                        dataset=config["dataset"],
                        data_dir=config['Data_dir'],
                        external_dataset=config["external_dataset"],
                        data_dir_ex=config["Data_dir_ex"],
                        seed=config["seed"],
                        model_name='cnn',
                        metric='accuracy',
                        device=gpu_index,
                        process=process)

    if fold_index == -1:
        cnn.validate()
    else:
        os.makedirs(f"checkpoint_dir/cnn_exp{fold_index}", exist_ok=True)
        cnn.cross_validation(fold_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to the .json configuration')
    args = parser.parse_args()

    # Read related parameters
    config = read_json(args.config)
    folds = config["folds"]
    gpu_idx = config["gpu_idx"]
    process = config["process"]

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])

    # to perform CNN training and testing
    for fold in folds:
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        run_name = f"fold{fold}_{config['process']}_{config['dataset']}_{dt_string}"
        wandb.init(project="adni-mtl", name=run_name, reinit=True, config=config)
        cnn_main(config, process, gpu_idx, fold)
