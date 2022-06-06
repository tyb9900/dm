import sys
import os
import papermill as pm
import scrapbook as sb
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from libs.timer import Timer
from libs.lightgcn import LightGCN
from libs.ImplicitCF import ImplicitCF
from libs.python_splitters import python_stratified_split
from libs.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from libs.constants import SEED as DEFAULT_SEED
from libs.deeprec_utils import prepare_hparams

yaml_file = "../libs/lightgcn.yaml"


df = pd.read_json('dataset.json')

df = df.drop(columns=['title','text','property_dict'])

print(df.head())

train, test = python_stratified_split(df, ratio=0.75)
data = ImplicitCF(train=train, test=test, seed=DEFAULT_SEED)

print(data)

print("HELLO")