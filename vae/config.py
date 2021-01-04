# encoding: utf-8
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""
import os
import torch

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Local directory of CypherCat API
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(ROOT_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')
Brain_DIR = os.path.join(DATASETS_DIR, 'brain')

RESULT_DIR = os.path.join(REPO_DIR, 'results')
# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

# difference datasets config
# train_batch_size, latent_dim, all_data_size, train_lr
DATA_PARAMS = {
    'adhd': (64, 10, 60000, 1e-3),
}
