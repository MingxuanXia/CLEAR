import torch
import numpy as np
import utils
import math
import torch.nn as nn
from utils import *

def test(args, X_test, model):
	X_test_tensor = torch.from_numpy(X_test).float().cuda().detach()
	iter_per_epoch = int(math.ceil(X_test_tensor.shape[0] / args.batch_size))
	Y_pred = []
	with torch.no_grad():
		for i in range(iter_per_epoch):
			start_idx = (i * args.batch_size) % X_test_tensor.shape[0]
			X_batch = X_test_tensor[start_idx: start_idx + args.batch_size, :]
			Y_pred_batch = model.forward(X_batch)
			Y_pred += [Y_pred_batch.detach().cpu().numpy()]
	Y_pred = np.concatenate(Y_pred, axis=0) 
	return Y_pred

def estimate_noise_rate(model, data, optimizer_es, args, true_tm, no_verbose=True):
    print('Estimate transition matirx......Waiting......')
    
    X_train, Y_train, Y_P_train, X_test, Y_test = data
    batch_size = args.batch_size
    iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
    
    A = torch.zeros((args.nc, args.es_epochs, len(X_train), 2))
    
    for epoch in range(args.es_epochs):
      
        # shuffle indices
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)
        
        print('epoch {}'.format(epoch + 1))
        model.train()
        train_loss = 0.
     
        for i in range(iter_per_epoch):
            optimizer_es.zero_grad()

            start_idx = (i * batch_size) % X_train.shape[0]
            idx = train_indicies[start_idx: start_idx + batch_size]
            batch_x = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
            batch_y = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
            
            out = model(batch_x)
            loss = torch.nn.BCELoss()(out, batch_y.float())
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer_es.step()
            
        print('Train Loss: {:.6f}'.format(train_loss / (len(X_train)) * batch_size))

        # testing
        if (epoch+1) % 5 == 0:
            Y_pred = test(args, X_test, model)
            f1_S, f1_O, f1_C = evaluate(Y_test, Y_pred)
            if not no_verbose:
                print('Warmup Epoch %d: r_loss %.4f, h_loss %.4f, ap %.4f' % (epoch, f1_S, f1_O, f1_C))
        
        with torch.no_grad():
            model.eval()
            for i in range(iter_per_epoch):
                start_idx = (i * batch_size) % X_train.shape[0]
                idx = train_indicies[start_idx: start_idx + batch_size]
                batch_x = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
                
                out = model(batch_x)
                out = out.cpu()
                for j in range(args.nc):
                    out_j = torch.stack([1 - out[:, j], out[:, j]]).t()
                    A[j, epoch, idx, :] = out_j
           
    # estimation
    True_T = true_tm
    error_abs = 0
    est_T = np.zeros_like(True_T)
    for i in range(args.nc):
        model_index = args.es_epochs - 1
        prob_ = A[i]
        transition_matrix_ = utils.fit(prob_[model_index, :, :], 2) 
        transition_matrix = utils.norm(transition_matrix_)
        
        T = transition_matrix
        if T[0, 1] + T[1, 0] >= 1.0:
            temp = T[0, 1] + T[1, 0]
            T[0, 1] = (T[0, 1] / temp) * 0.9
            T[1, 0] = (T[1, 0] / temp) * 0.9
            T[0, 0] = 1 - T[0, 1]
            T[1, 1] = 1 - T[1, 0]     
        
        estimate_error_abs = utils.error_abs(T, True_T[i])
        est_T[i] = T
        error_abs += estimate_error_abs
        print('class', i, 'estimation', T[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]])
    print('mean abs error', error_abs / args.nc)

    # estimate using Sk
    for topk in [1, 20]:
        print('Estimate using Sk top:', topk)
        True_T = true_tm
        error_abs = 0
        est_T = np.zeros_like(True_T)
        for i in range(args.nc):
            model_index = args.es_epochs - 1
            prob_ = A[i]
            transition_matrix_ = utils.fit_s(prob_[model_index, :, :], 2, Y_P_train[:, i], topk=topk)
            transition_matrix = utils.norm(transition_matrix_)
            
            T = transition_matrix
            if T[0, 1] + T[1, 0] >= 1.0:
                temp = T[0, 1] + T[1, 0]
                T[0, 1] = (T[0, 1] / temp) * 0.9
                T[1, 0] = (T[1, 0] / temp) * 0.9
                T[0, 0] = 1 - T[0, 1]
                T[1, 1] = 1 - T[1, 0]   

            estimate_error_abs = utils.error_abs(T, True_T[i])
            est_T[i] = T
            error_abs += estimate_error_abs
            print('class', i, 'estimation', T[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]])
        print('mean abs error', error_abs / args.nc)
    
    return est_T