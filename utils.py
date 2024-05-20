import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics import f1_score
import numpy as np
import os

LOG_EPSILON = 1e-5

def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def error_abs(T, T_true):
    error = np.sum(np.abs(T-T_true))
    return error

def fit(X, num_classes, filter_outlier=False):
    # print(X)  # [num_samples, 2]
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T

def fit_s(X, num_classes, S, topk=1):
    # X:[num_samples, 2]; S:[num_samples, 1]
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        idx_best = np.argsort(eta_corr[:, i])[-topk:]  
        T[i, 0] = (1 - S)[idx_best].mean()
        T[i, 1] = S[idx_best].mean()       
    return T    

def convert_labels(Y):
    Y = (Y > 0) * 1  # convert to 0/1 labels
    return Y

def gen_crowdsourced_label(Y, num_anno, noise_rate=None):
    Y_cs = np.copy(Y)
    Y_cs = Y_cs.reshape(1, Y.shape[0], Y.shape[1])
    Y_cs = Y_cs.repeat(num_anno, axis=0)
        
    N, nc = Y.shape
    for m in range(num_anno):
        rand_mat = np.random.rand(N, nc)
        mask = np.zeros((N, nc), dtype=np.float)
        for j in range(nc):
            yj = Y[:, j]
            mask[yj!=1, j] = rand_mat[yj!=1, j] < noise_rate[2*m]  # p of 0->1
            mask[yj==1, j] = rand_mat[yj==1, j] < noise_rate[2*m+1]  # p of 1->0
        Y_cs[m][mask==1] = (1-Y[mask==1])
        # real noise rate
        for i in range(nc):
            noise_rate_p = sum(Y_cs[m][Y[:,i]==1, i]==0) / sum(Y[:, i]==1)  # 1->0
            noise_rate_n = sum(Y_cs[m][Y[:,i]==0, i]==1) / sum(Y[:, i]==0)  # 0->1
            print('Annotator', str(m), 'class', str(i), 'noise_rate_n', round(noise_rate_n, 3), 
                  'noise_rate_p', round(noise_rate_p, 3), 'n', sum(Y[:, i]==0), 'p', sum(Y[:, i]==1))
    # real noise rate for soft label
    Y_mv = np.mean(Y_cs, axis=0)  # using soft label
    # Y_mv = (Y_mv >= 0.5) * 1
    print(Y_mv.shape)
    real_tm = np.zeros((nc, 2, 2))
    for i in range(nc):
        noise_rate_p = 1 - sum(Y_mv[Y[:,i]==1, i]) / sum(Y[:, i]==1)
        noise_rate_n = sum(Y_mv[Y[:,i]==0, i]) / sum(Y[:, i]==0)
        print('Class ', str(i), 'total noise_rate_n', round(noise_rate_n, 3), 'noise_rate_p', round(noise_rate_p, 3))
        real_tm[i, 1, 1] = 1 - noise_rate_p
        real_tm[i, 1, 0] = noise_rate_p
        real_tm[i, 0, 1] = noise_rate_n
        real_tm[i, 0, 0] = 1 - noise_rate_n

    Y_cs = np.transpose(Y_cs, (1, 0, 2))
    print('Data size Y_cs: ', Y_cs.shape)

    return Y_cs

def load_data_k_fold_cmll(path, num_anno, noise_rate=None, k=10, algo=None):
    
    n1 = np.array([noise_rate[2*i] for i in range(int(len(noise_rate) / 2))]).mean()
    n2 = np.array([noise_rate[2*i+1] for i in range(int(len(noise_rate) / 2))]).mean()
    mat_path = path + f'_M{num_anno}T0{(n1 * 10):.0f}0{(n2 * 10):.0f}.mat'
    print(noise_rate, n1, n2)

    if not os.path.exists(mat_path):
        print('Generating noisy .mat data ...')
        data = sio.loadmat(path)
        # preprocessing
        X = data['data']
        X = preprocessing.scale(X)
        Y = data['target']
        if X.shape[0] != Y.shape[0] and X.shape[0] == Y.shape[1]:
            Y = Y.T
        print('Data size (X, Y): ', X.shape, Y.shape)

        # generating cmll data
        Y = convert_labels(Y)
        Y_cs = gen_crowdsourced_label(Y, num_anno, noise_rate)
        assert ((Y_cs==1).sum() + (Y_cs==0).sum()) == (Y_cs.shape[0] * Y_cs.shape[1] * Y_cs.shape[2])

        # shuffling data
        data_num = int(X.shape[0])
        perm = np.arange(data_num)
        np.random.shuffle(perm)
        X = X[perm]
        Y = Y[perm]
        Y_cs = Y_cs[perm]

        print(f'Saving noisy {mat_path}, X/Y/Y_cs shape:', X.shape, Y.shape, Y_cs.shape)
        sio.savemat(mat_path, {'data': X, 'target': Y, 'pLabels': Y_cs})
    
    else:
        mat_data = sio.loadmat(mat_path)
        X = mat_data['data']
        Y = mat_data['target']
        Y_cs = mat_data['pLabels']
        print(f'Loading noisy {mat_path}, X/Y/Y_cs shape:', X.shape, Y.shape, Y_cs.shape)
        data_num = int(X.shape[0])
    
    # splitting data
    X_folds = [None] * k
    Y_folds = [None] * k
    Ycs_folds = [None] * k
    for i in range(k):
        start = int(data_num * (1.0 * i / k))
        if i < k - 1:
            end = int(data_num * (1.0 * (i+1) / k))
        else:
            end = data_num
        X_folds[i] = X[start:end, :]
        Y_folds[i] = Y[start:end, :]
        Ycs_folds[i] = Y_cs[start:end, :]

    return (X_folds, Y_folds, Ycs_folds)

def evaluate(Y_test, scores, threshold=0.5):
    return metric(Y_test, 1 * (scores > threshold), scores)

def metric(y, y_pred, scores):
    f1_S = f1_score(y, y_pred, average='samples')
    f1_O = f1_score(y, y_pred, average='micro')
    f1_C = f1_score(y, y_pred, average='macro')
    return f1_S, f1_O, f1_C