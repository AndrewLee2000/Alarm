import argparse
import numpy as np
import os
import torch
import json
from tqdm import tqdm
import time
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs import args_point_robot, args_half_cheetah_vel, args_half_cheetah_dir, args_ant_dir, args_walker, args_hopper, args_reach
from decision_transformer.model import DecisionTransformer
from decision_transformer.trainer import DT_Trainer
from decision_transformer.dataset import DT_Dataset, convert_data_to_trajectories
from decision_transformer.evaluation import evaluate_episode_rtg
from src.envs import PointEnv, HalfCheetahVelEnv, HalfCheetahDirEnv, AntDirEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', type=str, default='ant_dir')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--single_task', action='store_true', help='Train on single task only')
parser.add_argument('--task_id', type=int, default=0, help='Task ID for single task training')
args, rest_args = parser.parse_known_args()
env_type = args.env_type
single_task = args.single_task
task_id_single = args.task_id
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')