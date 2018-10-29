import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
random.seed(12796)

#LOAD DATA

sent1_train = []
sent2_train = []
labels_train = []
with open('hw2_data/snli_train.tsv') as train:
    snli_train = csv.reader(train, delimiter = '\t')
    for s1, s2, label in snli_train:
        sent1_train.append(s1.split())
        sent2_train.append(s2.split())
        if label == 'contradiction':
            labels_train.append(0)
        if label == 'neutral':
            labels_train.append(1)
        if label == 'entailment':
            labels_train.append(2)
    sent1_train.pop(0)
    sent2_train.pop(0)

sent1_val = []
sent2_val = []
labels_val = []
with open('hw2_data/snli_val.tsv') as val:
    snli_val = csv.reader(val, delimiter = '\t')
    for s1, s2, label in snli_val:
        sent1_val.append(s1.split())
        sent2_val.append(s2.split())
        if label == 'contradiction':
            labels_val.append(0)
        if label == 'neutral':
            labels_val.append(1)
        if label == 'entailment':
            labels_val.append(2)
    sent1_val.pop(0)
    sent2_val.pop(0)
    
#LOAD FASTTEXT WORD VECTORS

FastText = []
with open('wiki-news-300d-1M.vec', "r") as ft:
    for i, line in enumerate(ft):
        if i == 0:
            continue
        FastText.append(line)
        if i == 50000:
            break
