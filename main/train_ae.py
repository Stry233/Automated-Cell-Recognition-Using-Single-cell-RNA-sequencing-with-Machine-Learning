from autoencoder import AutoEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def getData(train):
    if train:
        path = 'data/train_data.h5'
        mode = 'train'
    else:
        path = 'data/test_data.h5'
        mode = 'test'

    # Load the data and convert to numpy array
    print("Start loading the data")
    store = pd.HDFStore(path)
    ori_rpkm = store['rpkm']
    samples = []
    cnt = 1
    total = len(ori_rpkm.index)
    for idx in ori_rpkm.index:
        print(f'\r{cnt} / {total} finished', end = "")
        samples.append(ori_rpkm.loc[idx].tolist())
        cnt += 1
    samples = np.array(samples)
    print('\nFinish loading the data')

    # Convert the label to index and save the corresponding index and cell type
    print("Start converting the labels")
    ori_labels = store['labels']
    if train:
        labels = []
        for idx in ori_labels.index:
            labels.append(ori_labels[idx])
        res_labels = []
        idx_name_dict = {}
        cnt = 0
        for l in labels:
            if l not in idx_name_dict.keys():
                idx_name_dict[l] = cnt
                res_labels.append(cnt)
                cnt += 1
            else:
                res_labels.append(idx_name_dict[l])
        with open(f'data/{mode}_idx_name.txt', 'w') as f:
            for key in idx_name_dict.keys():
                f.write(key)
                f.write(';')
                f.write(str(idx_name_dict[key]))
                f.write('\n')
    else:
        idx_name_dict = {}
        res_labels = []
        with open('./data/train_idx_name.txt', 'r') as f:
            line = f.readline()
            while line:
                arr = line.split(';')
                if len(arr) == 1:
                    break
                idx_name_dict[arr[0]] = int(arr[1])
                line = f.readline()
        for n in ori_labels:
            res_labels.append(idx_name_dict[n])
    print("Finish converting the labels")
    store.close()
    del store
    return samples, res_labels


class BioDataset(Dataset):
    def __init__(self, train):
        X, self.y = getData(train)
        scaler = StandardScaler().fit(X)
        self.X = torch.FloatTensor(scaler.transform(X))

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)

train_set = BioDataset(True)
test_set = BioDataset(False)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

device = 'cuda'
ae = AutoEncoder(20499, 3)
ae = ae.to(device)
default_lr = 1e-3
max_epoch = 50
loss_list = []
epoch_list = []
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=default_lr)

for epoch in range(max_epoch):
    print(epoch, "/", max_epoch, 'epoches finished')
    total_loss = 0
    cnt = 0
    t = tqdm(train_loader)
    t.set_description("epoch: %s" % epoch)
    for data in t:
        optimizer.zero_grad()
        samples, labels = data
        samples = samples.to(device)
        labels = labels.to(device)
        encoded, decoded = ae(samples)
        loss = loss_func(decoded, samples)
        total_loss += loss
        cnt += 1
        loss.backward()
        t.set_postfix(loss=loss.item())
        optimizer.step()
    # epoch_list.append(epoch + 1)
    # loss_list.append(total_loss / cnt)
    print('Average loss:', total_loss / cnt)

plt.plot(epoch_list, loss_list, '.-', color='b', label='Loss')
plt.legend()
plt.savefig('resultAndScore/AE_Loss.png')
# save the model for testing
def write_result(DR_method, clf, accuracy):
    with open(f'./resultAndScore/Result_new2.txt', 'a') as f:
        f.write(f'Dimension Reduction: {DR_method}\n')
        f.write(f'Classifier: {clf}\n')
        f.write("Accuracy: %.4f" % (100*accuracy))
        f.write('%')
        f.write('\n')
        f.write('-------------------------------------------------------------------\n')

X_train = []
y_train = []
for idx, data in enumerate(train_loader):
    samples, labels = data
    samples = samples.to(device)
    for l in labels:
        y_train.append(l)
    encoded, _ = ae(samples)
    for sample in encoded.tolist():
        X_train.append(sample)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
for idx, data in enumerate(test_loader):
    samples, labels = data
    samples = samples.to(device)
    for l in labels:
        y_test.append(l)
    encoded, _ = ae(samples)
    for sample in encoded.tolist():
        X_test.append(sample)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

with open('resultAndScore/AE_3d.txt', 'w') as f:
    for i in X_train:
        f.write(str(i[0]))
        f.write(' ')
        f.write(str(i[1]))
        f.write(' ')
        f.write(str(i[2]))
        f.write('\n')

with open('resultAndScore/AE_3d_labels.txt', 'w') as f:
    for i in y_train:
        f.write(str(i))
        f.write(' ')

# clf = RandomForestClassifier()
clf = SVC()
clf.fit(X_train, y_train)
write_result(f'AutoEncoder(n={X_train.shape[1]})', 'RandomForest', clf.score(X_test, y_test))
