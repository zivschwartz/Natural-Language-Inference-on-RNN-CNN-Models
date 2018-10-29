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

# LOAD DATA

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
    
# LOAD FASTTEXT WORD VECTORS

FastText = []
with open('wiki-news-300d-1M.vec', "r") as ft:
    for i, line in enumerate(ft):
        if i == 0:
            continue
        FastText.append(line)
        if i == 50000:
            break

# EMBEDDING FUNCTION 

def build_embedding(data):    
    word2id = {"<pad>": 0, "<unk>": 1}
    id2word = {0: "<pad>", 1: "<unk>"}
    embeddings = [
        np.zeros(300),
        np.random.normal(0, 0.01, 300),
    ]
    
    for i, line in enumerate(data, start=2):
        parsed = line.split()
        word = parsed[0]
        array = np.array([float(x) for x in parsed[1:]])
    
        word2id[word] = i
        id2word[i] = word
        embeddings.append(array)
    
    return word2id, id2word, embeddings
 
word2id, id2word, embeddings = build_embedding(FastText)

MAX_SENT_LENGTH = max(max([len(sent) for sent in sent1_train]), max([len(sent) for sent in sent2_train]))
BATCH_SIZE = 32
PAD_IDX = 0
UNK_IDX = 1

# PYTORCH DATALOADER
class VocabDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, sent1_ls, sent2_ls, labels_ls, word2id):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.sent1_ls = sent1_ls
        self.sent2_ls = sent2_ls
        self.labels_ls = labels_ls
        assert len(sent1_ls) == len(sent2_ls)
        assert len(sent1_ls) == len(labels_ls)
        self.word2id = word2id

    def __len__(self):
        return len(self.sent1_ls)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        word_idx_1 = [
            self.word2id[word] if word in self.word2id.keys() else UNK_IDX  
            for word in self.sent1_ls[key][:MAX_SENT_LENGTH]
        ]
        word_idx_2 = [
            self.word2id[word] if word in self.word2id.keys() else UNK_IDX  
            for word in self.sent2_ls[key][:MAX_SENT_LENGTH]
        ]
        label = self.labels_ls[key]
        return [word_idx_1, word_idx_2, len(word_idx_1), len(word_idx_2), label]

def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    sent1_ls = []
    sent2_ls = []
    length_sent1_ls = []
    length_sent2_ls = []
    labels_ls = []
    
    for datum in batch:
        labels_ls.append(datum[4])
        length_sent1_ls.append(datum[2])
        length_sent2_ls.append(datum[3])
        
    for datum in batch:
        padded_vec1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_SENT_LENGTH-datum[2])),
                                mode="constant", constant_values=0)
        sent1_ls.append(padded_vec1)
        padded_vec2 = np.pad(np.array(datum[1]),
                                pad_width=((0,MAX_SENT_LENGTH-datum[3])),
                                mode="constant", constant_values=0)
        sent2_ls.append(padded_vec2)
    
    sent1_ls = np.array(sent1_ls)
    sent2_ls = np.array(sent2_ls)
    labels_ls = np.array(labels_ls)
    
    return [
        torch.from_numpy(np.array(sent1_ls)), 
        torch.from_numpy(np.array(sent2_ls)),
        torch.LongTensor(length_sent1_ls),
        torch.LongTensor(length_sent2_ls),
        torch.LongTensor(labels_ls),
    ]

# BUILD TRAINING AND VALIDATION DATALOADERS

train_dataset = VocabDataset(sent1_train, sent2_train, labels_train, word2id)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True)

val_dataset = VocabDataset(sent1_val, sent2_val, labels_val, word2id)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True)

# RNN MODEL

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size):
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        
        #Implementing a Bidirectional GRU
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        #Hidden Size * 4, since we have two sentences per instance and it is bidirectional
        self.linear = nn.Linear(4*hidden_size, num_classes) 

    def init_hidden(self, batch_size):

        hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size) ## time 2
        return hidden

    def forward(self, x1, x2, lenx1, lenx2):

        x1desc_idx = np.argsort(np.array(lenx1))[::-1]
        x2desc_idx = np.argsort(np.array(lenx2))[::-1]
        
        sorted_lenx1 = (np.array(lenx1)[x1desc_idx])
        sorted_lenx2 = (np.array(lenx2)[x2desc_idx])
        
        x1order = (np.linspace(0, len(lenx1), len(lenx1), endpoint = False)).astype('int')
        x2order = (np.linspace(0, len(lenx2), len(lenx2), endpoint = False)).astype('int')
        
        batch_size1, seq_len1 = x1.size()
        batch_size2, seq_len2 = x2.size()

        self.hidden1 = self.init_hidden(batch_size1)
        self.hidden2 = self.init_hidden(batch_size2)
        
        embed1 = self.embedding(x1)
        embed2 = self.embedding(x2)
        
        embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, sorted_lenx1, batch_first=True)
        embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, sorted_lenx2, batch_first=True)
        
        x1reorder = (x1order[x1desc_idx])
        x2reorder = (x2order[x2desc_idx])
        
        reversed1 = np.argsort(x1reorder)
        reversed2 = np.argsort(x2reorder)
        
        rnn_out1, self.hidden1 = self.rnn(embed1, self.hidden1)
        rnn_out2, self.hidden2 = self.rnn(embed2, self.hidden2)
        
        rnn_out1, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out1, batch_first=True)
        rnn_out2, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out2, batch_first=True)

        rnn_out1 = rnn_out1[reversed1]
        rnn_out2 = rnn_out2[reversed2]
        
        rnn_out1 = torch.sum(rnn_out1, dim=1)
        rnn_out2 = torch.sum(rnn_out2, dim=1)
        
        full_rnn_out = torch.cat([rnn_out1, rnn_out2], dim=1)
        logits = self.linear(full_rnn_out)
        #ReLU
        logits1 = self.linear(full_rnn_out)
    
        return logits1
    
# TEST MODEL FUNCTION
    
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for x1, x2, lenx1, lenx2, labels in loader:
        x1_batch, x2_batch, lenx1_batch, lenx2_batch, label_batch = x1, x2, lenx1, lenx2, labels
        outputs = F.softmax(model(x1_batch, x2_batch, lenx1_batch, lenx2_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

model = RNN(emb_size=100, hidden_size=200, num_layers=1, num_classes=3, vocab_size=len(id2word))

learning_rate = 3e-4
num_epochs = 5 #Epoch size reduced to take into account size of data

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

# TRAIN MODEL
# TRAINING ACCURACY
# VALIDATION ACCURACY

rnn_val_acc = []
rnn_train_acc = []
for epoch in range(num_epochs):
    for i, (x1, x2, lenx1, lenx2, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        outputs = model(x1, x2, lenx1, lenx2)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if i > 0 and i % 100 == 0:
            val_acc = test_model(val_loader, model)
            rnn_val_acc.append(val_acc)
            train_acc = test_model(train_loader, model)
            rnn_train_acc.append(train_acc)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))

# PLOT TRAINING AND VALIDATION ACCURACIES FOR BASELINE RNN MODEL 

%matplotlib inline
plt.figure(figsize = (8,6))
plt.plot(rnn_val_acc, 'r', label = 'Validation Accuracy')
plt.plot(rnn_train_acc, 'b', label = 'Training Accuracy')
plt.title('Training and Validation Accuracy on RNN')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()
