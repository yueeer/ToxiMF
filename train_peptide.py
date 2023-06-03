import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader,TensorDataset
from sklearn import metrics
from sklearn.model_selection import train_test_split
import scipy.io as sio
from dvib_peptide import dvib
from label_smoothing import LabelSmoothing
import scipy.io as io

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

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


def make_tensor(sequences,labels):
    evolution = torch.zeros(len(sequences),max_seq_len,20)
    lengths = []
    for i in range(len(sequences)):
        lengths.append((len(sequences[i])))
        temp = get_blosum62(sequences[i])
        evolution[i,:,:] = torch.Tensor(temp)

    return evolution,torch.Tensor(lengths),torch.Tensor(labels)

max_seq_len = 50

data = pd.read_csv('dataset/peptide/data.csv')
labels = data['label'].values
sequences = data['sequence'].values
D_train, D_test, Dy_train, Dy_test = train_test_split(
    sequences, labels, test_size=0.15, random_state=76, stratify=labels)
train_pssm, train_len, train_label = make_tensor(D_train, Dy_train)
test_pssm, test_len, test_label = make_tensor(D_test, Dy_test)

# FEGS features
FEGS = sio.loadmat('dataset/peptide/data_CA.mat')['samples']
X_train, X_test, Xy_train, Xy_test = train_test_split(
    FEGS, labels, test_size=0.15, random_state=76, stratify=labels)
train_CA = torch.Tensor(X_train)
test_CA = torch.Tensor(X_test)

data_bert = sio.loadmat('dataset/peptide/esm_peptide_data.mat')['samples']
R_train, R_test, Ry_train, Ry_test = train_test_split(
    data_bert, labels, test_size=0.15, random_state=76, stratify=labels)
train_bert = torch.Tensor(R_train)
test_bert = torch.Tensor(R_test)

train_data = DataLoader(TensorDataset(train_pssm, train_len, train_CA, train_bert, train_label), batch_size=100, shuffle=True)
test_data = DataLoader(TensorDataset(test_pssm, test_len,test_CA, test_bert, test_label), batch_size=100)
print("data done")


def get_sn_sp_FDR(y_test, y_pre):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pre)
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    # Sensitivity
    Sn = tp / (tp + fn)
    # Specificity
    Sp = tn / (tn + fp)
    # False Discovery Rate
    FDR = fp / (tp + fp)

    return Sn, Sp, FDR

def get_auPRC(y_test, y_pre):
    precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_pre)
    auPRC = metrics.auc(recall, precision)
    return auPRC


out_channels = 128
hidden_size = 512
k = 1024

f1_list = []
min_test_f1 = 0
best_result = [0] * 8
test_f1 = {}

path = 'new_model/protein_final_k1024_channels128_hidden512.model'
model = dvib(k, out_channels, hidden_size, device).to(device)
model.load_state_dict(torch.load(path).get('model'))

optimizer = opt.Adam(model.parameters(), lr=0.00005)
LabelSmoothing = LabelSmoothing(size=2, smoothing=0.1)

num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0

    y_pre = []
    y_test = []
    train_blosum = []
    train_esm = []
    train_ca = []
    train_feature = []
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

        y_test.extend(label.cpu().detach().numpy())
        y_pre.extend(y_pred.argmax(dim=1).cpu().detach().numpy())

        train_blosum.extend(blosum.cpu().detach().numpy())
        train_esm.extend(esm.cpu().detach().numpy())
        train_ca.extend(ca.cpu().detach().numpy())
        train_feature.extend(allfeature.cpu().detach().numpy())

    print("Train Epoch:{} Average loss: {:.6f} ACC:{}/{} ({:.4f}%)".format(
        epoch,
        train_loss,
        train_correct, len(train_data.dataset),
        100. * train_correct / len(train_data.dataset)
    ))

    with open('result/peptide_ToxIBTL/train_predict.txt', "w") as f:
        for i in range(len(y_pre)):
            f.write(str(y_pre[i]) + '\n')
    f.close()
    with open('result/peptide_ToxIBTL/train_label.txt', "w") as q:
        for i in y_test:
            q.write(str(i) + '\n')
    q.close()
    # io.savemat('result/peptide_ToxIBTL/train_result_blosum.mat', {'samples': train_blosum})
    # io.savemat('result/peptide_ToxIBTL/train_result_esm.mat', {'samples': train_esm})
    # io.savemat('result/peptide_ToxIBTL/train_result_ca.mat', {'samples': train_ca})
    # io.savemat('result/peptide_ToxIBTL/train_result_feature.mat', {'samples': train_feature})

    correct = 0
    y_pre = []
    y_test = []
    x_length = []

    dev_blosum = []
    dev_esm = []
    dev_ca = []
    dev_feature = []
    with torch.no_grad():
        for batch_idx, (sequences, lengths, FEGS, bertF, labels) in enumerate(test_data):
            seq_lengths, perm_idx = lengths.sort(dim=0, descending=True)
            seq_tensor = sequences[perm_idx].to(device)
            FEGS_tensor = FEGS[perm_idx].to(device)
            label = labels[perm_idx].long().to(device)
            bertF_tensor = bertF[perm_idx].to(device)

            x_length.extend(seq_lengths.cpu().detach().numpy())

            y_test.extend(label.cpu().detach().numpy())

            y_pred = model(seq_tensor, seq_lengths, FEGS_tensor,bertF_tensor, device)

            y_pre.extend(y_pred.argmax(dim=1).cpu().detach().numpy())

            _, pred = torch.max(y_pred, 1)

            correct += pred.eq(label).sum().item()

            # dev_blosum.extend(blosum.cpu().detach().numpy())
            # dev_esm.extend(esm.cpu().detach().numpy())
            # dev_ca.extend(ca.cpu().detach().numpy())
            # dev_feature.extend(allfeature.cpu().detach().numpy())
        f1_list.append(metrics.f1_score(y_test, y_pre))
        sn, sp, FDR = get_sn_sp_FDR(y_test, y_pre)
        auPRC = get_auPRC(y_test, y_pre)
        print(
            '\nTest: Sn:({:.4f}%) Sp:({:.4f}%) Accuracy:{}/{} ({:.4f}%) f1:({:.4f}%) mcc:({:.4f}%) auc:({:.4f}) auPRC:({:.4f} FDR:({:.4f})\n'.format(
                sn,
                sp,
                correct, len(test_data.dataset),
                100. * correct / len(test_data.dataset),
                metrics.f1_score(y_test, y_pre),
                metrics.matthews_corrcoef(y_test, y_pre),
                metrics.roc_auc_score(y_test, y_pre),
                auPRC,
                FDR
            ))

        with open('result/peptide_ToxIBTL/dev_predict.txt', "w") as f:
            for i in range(len(y_pre)):
                f.write("> " + str(x_length[i]) + "\n")
                f.write(str(y_pre[i]) + '\n')
        f.close()
        with open('result/peptide_ToxIBTL/dev_label.txt',"w") as q:
            for i in y_test:
                q.write(str(i) + '\n')
        q.close()
        # io.savemat('result/peptide_ToxIBTL/dev_result_blosum.mat', {'samples': dev_blosum})
        # io.savemat('result/peptide_ToxIBTL/dev_result_esm.mat', {'samples': dev_esm})
        # io.savemat('result/peptide_ToxIBTL/dev_result_ca.mat', {'samples': dev_ca})
        # io.savemat('result/peptide_ToxIBTL/dev_result_feature.mat', {'samples': dev_feature})

        model_name = 'new_model/peptide_k' + str(k) + '_channels' + str(
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
            best_result[4], best_result[5], best_result[7] = get_sn_sp_FDR(y_test, y_pre)
            best_result[6] = get_auPRC(y_test, y_pre)
        print("----------------best------------------")
        print(
            '\nTest: Sn:({:.4f}%) Sp:({:.4f}%) Accuracy:({:.4f}%) f1:({:.4f}%) mcc:({:.4f}%) auc:({:.4f}) auPRC:({:.4f} FDR:({:.4f})\n'.format(
                best_result[4], best_result[5], best_result[0], best_result[1], best_result[2],
                best_result[3], best_result[6], best_result[7]
            ))

    final_model_name = 'new_model/peptide_final_k' + str(k) + '_channels' + str(
        out_channels) + '_hidden' + str(hidden_size) + '.model'
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'test_best_acc': f1_list[-1]},
               final_model_name)
    test_f1[(out_channels, hidden_size)] = max(f1_list)
