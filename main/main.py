import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from cuml import PCA
# from cuml.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from autoencoder import AutoEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import MDS
from sklearn.manifold import Isomap


def getParsing(inputRawData,mode):
    """
    This function will return the info based
    on input but exclues all unecessary info
    contain in the file
    
    Parameters
    ----------
    inputRawData : str
        the input file
        
    mode : str
        the mode (train, test, all)
    
    Returns
    -------
    dict
        key : str
            name of cell
        value : set(int)
            all id
    """
    # initialize all variable
    res = dict()
    nowKey = ""
    
    # delete all empty line then convert to list for traverse
    dataSplit = [x for x in inputRawData.split("\n") if x!='']
    
    # traverse start, input info in dict
    for line in dataSplit:
        lineSegment = line.split()
        idParsing = lineSegment[0].split(":")
        if (len(idParsing) == 2 and idParsing[0].isalpha() and idParsing[1].isdigit()):
            nowKey = " ".join(str(i) for i in lineSegment[1::])
        elif (len(lineSegment) == 3 and lineSegment[2].isdigit()):
            if (lineSegment[0][0] == '*' and (mode == "test" or mode == "all")): # case: test
                if (not nowKey in res.keys()):
                    res[nowKey] = set()
                res[nowKey].add(int(lineSegment[0][1::]))
            elif (lineSegment[0].isdigit() and (mode == "train" or mode == "all")): # case: train
                if (not nowKey in res.keys()):
                    res[nowKey] = set()
                res[nowKey].add(int(lineSegment[0]))
    return res


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
        print(f'\r{cnt} / {total} finished', end='')
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
    # Add normalization
    return samples, res_labels


def write_result(DR_method, clf, accuracy):
    with open(f'./resultAndScore/Result_final.txt', 'a') as f:
        f.write(f'Dimension Reduction: {DR_method}\n')
        f.write(f'Classifier: {clf}\n')
        f.write("Accuracy: %.4f" % (100*accuracy))
        f.write('%')
        f.write('\n')
        f.write('-------------------------------------------------------------------\n')


def tsne_test(X_train, X_test, y_train, y_test, saveRes, n):
    tsne = TSNE(n_components=n, n_jobs=-1)
    print('Start training TSNE')
    begin = time.time()
    data = np.vstack((X_train, X_test))
    print(data.shape)
    reduced = tsne.fit_transform(data)
    print('Finish training')
    print(f"Time: {time.time() - begin}s")
    train_reduced = reduced[0:len(y_train)]
    test_reduced = reduced[len(y_train):]
    if saveRes:
        with open('tsne_reduced.txt', 'w') as f:
            for i in reduced:
                f.write(str(i[0]))
                f.write(' ')
                f.write(str(i[1]))
                f.write(' ')
                f.write(str(i[2]))
                f.write('\n')
    print('Start training classifier')
    begin = time.time()
    rfc = RandomForestClassifier()
    rfc.fit(train_reduced, y_train)
    print('Finish training classifier')
    accuracy = rfc.score(test_reduced, y_test)
    print(accuracy)
    # getConfusionMat(y_test, rfc.predict(test_reduced), 'TSNE_confusion_mat')
    write_result('TSNE', 'Random forest(Default params)', accuracy)


def PCA_test(X_train, X_test, y_train, y_test, n, saveDRRes):
    # Dimension reduction
    # pca = PCA(n_components=n)
    mds = Isomap(n_components=n, n_jobs=-1)
    print('Start training PCA')
    begin = time.time()
    train_reduced = mds.fit_transform(X_train)
    print('Finish training')
    print(f"Time: {time.time() - begin}s")
    if saveDRRes:
        with open('resultAndScore/isomap_reduced_3d.txt', 'w') as f:
            for i in train_reduced:
                f.write(str(i[0]))
                f.write(' ')
                f.write(str(i[1]))
                f.write(' ')
                f.write(str(i[2]))
                f.write('\n')
    test_reduced = mds.transform(X_test)

    # Classification
    # m_l_list = []
    # acc_list = []
    # for m_l in range(1, 16):
    clf = RandomForestClassifier()
    # clf =  LinearSVC()
    # clf = GaussianNB()
    print(f'Start training classifier')
    begin = time.time()
    # clf.fit(X_train, y_train)
    clf.fit(train_reduced, y_train)
    print('Finish training classifier')
    accuracy = clf.score(test_reduced, y_test)
    # accuracy = clf.score(X_test, y_test)
    write_result(f'MDS(n={n})', f'{clf.__class__.__name__}', accuracy)
    # print('Finish All')
    # plt.plot(m_l_list, acc_list, 's-', color='b')
    # plt.xlabel('min_samples_leaf')
    # plt.ylabel('Accuracy')
    # plt.savefig('resultAndScore/PCA_RFC_ML_1.png')

def PCA_TSNE_test(X_train, X_test, y_train, y_test, saveRes, n):
    pca = PCA(n_components=n)
    data = np.vstack((X_train, X_test))
    pca_reduced = pca.fit_transform(data)
    print('PCA finished')
    begin = time.time()
    tsne = TSNE(n_components=3)
    reduced = tsne.fit_transform(pca_reduced)
    print(f'TSNE finished, use {time.time() - begin}s')
    train_reduced = reduced[0:len(y_train)]
    test_reduced = reduced[len(y_train):]
    if saveRes:
        with open('resultAndScore/pca_tsne_reduced_3d.txt', 'w') as f:
            for i in reduced:
                f.write(str(i[0]))
                f.write(' ')
                f.write(str(i[1]))
                f.write(' ')
                f.write(str(i[2]))
                f.write('\n')
    print('Start training classifier')
    # rfc = RandomForestClassifier()
    # rfc.fit(train_reduced, y_train)
    clf = SVC()
    clf.fit(train_reduced, y_train)
    print('Finish training classifier')
    # accuracy = rfc.score(test_reduced, y_test)
    accuracy = clf.score(test_reduced, y_test)
    print(accuracy)
    # getConfusionMat(y_test, rfc.predict(test_reduced), 'TSNE_confusion_mat')
    write_result(f'PCA(n={n}) + TSNE', 'SVC', accuracy)



def getConfusionMat(y_test, pred, title):
    con_mat = confusion_matrix(y_test, pred)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    con_mat_norm = np.around(con_mat_norm, decimals=2)

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm, annot=False, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'resultAndScore/{title}.png')


if __name__ == '__main__':
    X_train, y_train = getData(True)
    X_test, y_test = getData(False)
    X_train = np.float32(X_train)
    y_train = np.float32(y_train)
    X_test = np.float32(X_test)
    y_test = np.float32(y_test)

    PCA_test(X_train, X_test, y_train, y_test, 3, True)
    # This one is for testing PCA + t-SNE
    # False means save the TSNE result, should set to false
    # The last number is the dimension reduced by PCA
    # that is, 20499->n: PCA, n->2/3: TSNE
    # PCA_TSNE_test(X_train, X_test, y_train, y_test, False, 26)

    # This one is for do the PCA test
    # n is the dimension reduced by PCA
    # Should change the clf in PCA_test
    # for n in range(10, 81, 10):
    #     PCA_TSNE_test(X_train, X_test, y_train, y_test, False, n)
    #   PCA_test(X_train, X_test, y_train, y_test, n, False)
