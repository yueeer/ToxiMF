import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader,TensorDataset
from sklearn import metrics
import scipy.io as sio
from dvib_protein import dvib
from label_smoothing import LabelSmoothing

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

max_seq_len = 1002
dataset = "protein"
train_path = 'dataset/' + dataset + '/train.csv'
test_path = 'dataset/' + dataset + '/dev.csv'
# save_path = 'amp_Data/' + dataset + '/result.txt'
# label_path = 'amp_Data/' + dataset + '/label.txt'


blosum62 = {
        'A': [4, -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0, -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }

def get_blosum62(seq):
    blosum_list = []
    for i in seq:
        blosum_list.append(blosum62[i])
    blosum = np.array(blosum_list)
    feature = np.zeros((max_seq_len,20))
    idx = blosum.shape[0]
    feature[0:idx,:] = blosum
    return feature


def make_tensor(path):
    data = pd.read_csv(path)
    sequences = data['sequence'].values
    labels = data['label'].values
    evolution = torch.zeros(len(sequences),max_seq_len,20)
    lengths = []
    for i in range(len(sequences)):
        lengths.append((len(sequences[i])))
        temp = get_blosum62(sequences[i])
        evolution[i,:,:] = torch.FloatTensor(temp)

    return evolution,torch.Tensor(lengths),torch.Tensor(labels)


train_data = sio.loadmat('dataset/' + dataset +'/train_CA.mat')
train_FEGS = torch.FloatTensor(train_data['samples'])
test_data = sio.loadmat('dataset/' + dataset +'/dev_CA.mat')
test_FEGS = torch.FloatTensor(test_data['samples'])

train_pssm, train_len,train_label = make_tensor(train_path)
test_pssm, test_len,test_label = make_tensor(test_path)

train_bert_path = sio.loadmat('dataset/' + dataset + '/esm_protein_train.mat')
train_bert = torch.FloatTensor(train_bert_path['samples'])
test_bert_path = sio.loadmat('dataset/' + dataset + '/esm_protein_dev.mat')
test_bert = torch.FloatTensor(test_bert_path['samples'])

train_data = DataLoader(TensorDataset(train_pssm, train_len,train_FEGS, train_bert, train_label), batch_size=100, shuffle=True)
test_data = DataLoader(TensorDataset(test_pssm, test_len,test_FEGS, test_bert, test_label), batch_size=100)
print("data done")


def get_sn_sp(y_test, y_pre):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pre)
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    # Sensitivity
    Sn = tp / (tp + fn)
    # Specificity
    Sp = tn / (tn + fp)
    return Sn, Sp

def get_auPRC(y_test, y_pre):
    precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_pre)
    auPRC = metrics.auc(recall, precision)
    return auPRC

out_channels = 128
hidden_size = 512
k = 1024

f1_list = []
min_test_f1 = 0
best_result = [0] * 7
test_f1 = {}

model = dvib(k, out_channels, hidden_size, device).to(device)
optimizer = opt.Adam(model.parameters(), lr=0.0001)
LabelSmoothing = LabelSmoothing(size=2, smoothing=0.1)
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_idx, (sequences, lengths, FEGS, bertF, labels) in enumerate(train_data):
        seq_lengths, perm_idx = lengths.sort(dim=0, descending=True)
        seq_tensor = sequences[perm_idx].to(device)
        FEGS_tensor = FEGS[perm_idx].to(device)
        label = labels[perm_idx].long().to(device)
        bertF_tensor = bertF[perm_idx].to(device)

        y_pred = model(seq_tensor, seq_lengths, FEGS_tensor, bertF_tensor, device)
        loss = LabelSmoothing(nn.LogSoftmax(dim=-1)(y_pred), label)

        _, train_pred = torch.max(y_pred, 1)
        train_correct += train_pred.eq(label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print("Train Epoch:{} Average loss: {:.6f} ACC:{}/{} ({:.4f}%)".format(
        epoch,
        train_loss,
        train_correct, len(train_data.dataset),
        100. * train_correct / len(train_data.dataset)
    ))

    correct = 0
    y_pre = []
    y_test = []
    with torch.no_grad():
        for batch_idx, (sequences, lengths, FEGS, bertF, labels) in enumerate(test_data):
            seq_lengths, perm_idx = lengths.sort(dim=0, descending=True)
            seq_tensor = sequences[perm_idx].to(device)
            FEGS_tensor = FEGS[perm_idx].to(device)
            label = labels[perm_idx].long().to(device)
            bertF_tensor = bertF[perm_idx].to(device)

            y_test.extend(label.cpu().detach().numpy())

            y_pred = model(seq_tensor, seq_lengths, FEGS_tensor, bertF_tensor, device)
            y_pre.extend(y_pred.argmax(dim=1).cpu().detach().numpy())

            _, pred = torch.max(y_pred, 1)

            correct += pred.eq(label).sum().item()

        f1_list.append(metrics.f1_score(y_test, y_pre))
        sn, sp = get_sn_sp(y_test, y_pre)
        auPRC = get_auPRC(y_test, y_pre)
        print(
            '\nTest: Sn:({:.4f}%) Sp:({:.4f}%) Accuracy:{}/{} ({:.4f}%) f1:({:.4f}%) mcc:({:.4f}%) auc:({:.4f}) auPRC:({:.4f})\n'.format(
                sn,
                sp,
                correct, len(test_data.dataset),
                100. * correct / len(test_data.dataset),
                metrics.f1_score(y_test, y_pre),
                metrics.matthews_corrcoef(y_test, y_pre),
                metrics.roc_auc_score(y_test, y_pre),
                auPRC
            ))

        # with open(save_path, "w") as f:
        #     for i in y_pre:
        #         f.write(str(i) + '\n')
        # f.close()
        # with open(label_path,"w") as q:
        #     for i in y_test:
        #         q.write(str(i) + '\n')
        # q.close()
        #
        # score(save_path, label_path)

        model_name = 'new_model/protein_k' + str(k) + '_channels' + str(
            out_channels) + '_hidden' + str(hidden_size) + '.model'
        if (float(f1_list[-1]) > min_test_f1):
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'test_best_acc': f1_list[-1]},
                       model_name)  # 保存字典对象，里面'model'的value是模型
            min_test_f1 = float(f1_list[-1])
            best_result[0] = 100. * correct / len(test_data.dataset)
            best_result[1] = metrics.f1_score(y_test, y_pre)
            best_result[2] = metrics.matthews_corrcoef(y_test, y_pre)
            best_result[3] = metrics.roc_auc_score(y_test, y_pre)
            best_result[4], best_result[5] = get_sn_sp(y_test, y_pre)
            best_result[6] = get_auPRC(y_test, y_pre)
        print("----------------best------------------")
        print(
            '\nTest: Sn:({:.4f}%) Sp:({:.4f}%) Accuracy:({:.4f}%) f1:({:.4f}%) mcc:({:.4f}%) auc:({:.4f}) auPRC:({:.4f})\n'.format(
                best_result[4], best_result[5], best_result[0], best_result[1], best_result[2],
                best_result[3], best_result[6]
            ))

    final_model_name = 'new_model/protein_final_k' + str(k) + '_channels' + str(
        out_channels) + '_hidden' + str(hidden_size) + '.model'
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'test_best_acc': f1_list[-1]},
               final_model_name)
    test_f1[(out_channels, hidden_size)] = max(f1_list)
