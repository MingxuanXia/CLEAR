import numpy as np
import math
import warnings
import argparse
import torch
import random
from utils import *
from models import *
import torch.backends.cudnn as cudnn
from estimation import estimate_noise_rate

warnings.filterwarnings("ignore")

def run_model(args, data, model, optimizer, t_m, eval_every=5, scheduler=None):
	
	X_train, Y_train, Y_cs_train, X_test, Y_test = data
	batch_size = args.batch_size
	iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))

	t_m = torch.Tensor(t_m).float().cuda()
	Y_ema = Y_cs_train.copy()

	for e in range(args.epochs):
		# shuffle indices
		train_indicies = np.arange(X_train.shape[0])
		np.random.shuffle(train_indicies)
		
		model.train()

		for i in range(iter_per_epoch):
			start_idx = (i * batch_size) % X_train.shape[0]
			idx = train_indicies[start_idx: start_idx + batch_size]
			input_feat = torch.from_numpy(X_train[idx, :]).float().detach().cuda()  # x
			input_label = torch.from_numpy(Y_cs_train[idx, :]).float().detach().cuda()  # s
			input_label_e = torch.from_numpy(Y_ema[idx, :]).float().detach().cuda()  # s'

			# forward
			label_out, label_mu, label_logvar, label_z, feat_out, feat_mu, feat_logvar, feat_z = \
				model(input_label_e, input_feat)
			# loss
			total_loss, nll_loss, nll_loss_x, kl_loss, indiv_prob = \
				compute_loss(input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar, 
				  	label_z, feat_z, model.r_sqrt_sigma, args, args.n_train_sample, t_m)
			# update s'
			Y_ema[idx, :] = args.w_ema * Y_ema[idx, :] + (1 - args.w_ema) * indiv_prob.cpu().detach().numpy()

			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			if scheduler != None:
				scheduler.step()

		if args.no_verbose:
			eval_every = 1

		if (e+1) % eval_every == 0:
			Y_pred = test(args, X_test, Y_test, model, t_m)
			assert Y_pred.shape == Y_test.shape

			f1_S, f1_O, f1_C = evaluate(Y_test, Y_pred)
			print(f'Epoch {e+1}, Example-F1 {f1_S:.4f}, Micro-F1 {f1_O:.4f}, Macro-F1 {f1_C:.4f}')
	
def test(args, X_test, Y_test, model, t_m):
	'''
		Y_test only used for calling forward, no influence to the results of prediction
	'''
	X_test_tensor = torch.from_numpy(X_test).float().cuda().detach()
	Y_test_tensor = torch.from_numpy(Y_test).float().cuda().detach()
	iter_per_epoch = int(math.ceil(X_test_tensor.shape[0] / args.batch_size))
	Y_pred = []
	model.eval()
	with torch.no_grad():
		for i in range(iter_per_epoch):
			start_idx = (i * args.batch_size) % X_test_tensor.shape[0]
			X_batch = X_test_tensor[start_idx: start_idx + args.batch_size, :]
			Y_batch = Y_test_tensor[start_idx: start_idx + args.batch_size, :]
			label_out, label_mu, label_logvar, label_z, feat_out, feat_mu, feat_logvar, feat_z = model(Y_batch, X_batch)
			total_loss, nll_loss, nll_loss_x, kl_loss, indiv_prob = \
				compute_loss(Y_batch, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar, 
				  	label_z, feat_z, model.r_sqrt_sigma, args, args.n_test_sample, t_m)
			Y_pred += [indiv_prob.cpu().data.numpy()]
	Y_pred = np.concatenate(Y_pred, axis=0)
	return Y_pred


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch Implementation of CLEAR')
	# optimization parameters
	parser.add_argument("--epochs", default=200, type=int, 
					 help='max epoch to train')
	parser.add_argument("--es_epochs", default=100, type=int, 
					 help="epochs for transition estimation")
	parser.add_argument('--batch-size', default=32, type=int, metavar='N',
					 help='train batchsize')
	parser.add_argument("--lr", default=0.00075, type=float, 
					 help='initial learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
					 help='SGD momentum (default: 0.9)')
	parser.add_argument("--weight_decay", default=1e-5, type=float, 
					 help='weight decay rate')
	parser.add_argument("--lr_decay_ratio", default=0.5, type=float, 
					 help='The decay ratio of learning rate')
	parser.add_argument("--lr_decay_times", default=4.0, type=float, 
					 help='The number of times learning rate decays')
	# dataset parameters
	parser.add_argument("--noise_rate", type=str, default='0.1,0.1', 
					 help='[T01,T10]*M, for each annotator on all classes')
	parser.add_argument("--dataset", default='Image', type=str, 
					 help='MLL dataset')
	parser.add_argument('--tr_rate', type=float, default=0.9, 
					 help='training set ratio')
	# vae parameters
	parser.add_argument("--loss_type", default='unbiased', type=str, 
					 choices=['bce', 'unbiased'])
	parser.add_argument("--n_test_sample", default=10000, type=int, 
					 help='The sampling times for the testing')
	parser.add_argument("--n_train_sample", default=100, type=int, 
					 help='The sampling times for the training')
	parser.add_argument("--latent_dim", default=50, type=int, 
					 help='Gaussian subspace dimension')
	parser.add_argument("--nll_coeff", default=0.5, type=float, 
					 help='nll_loss coefficient')
	parser.add_argument("--d_coeff", default=1.0, type=float, 
					 help='distill_loss coefficient')
	parser.add_argument("--scale_coeff", default=1.0, type=float, 
					 help='mu/logvar scale coefficient')
	parser.add_argument("--keep_prob", default=0.5, type=float, 
					 help='drop out rate')
	parser.add_argument("--w_ema", type=float, default=0.9, 
					 help="momentum parameter of updating s'")
	# other parameters
	parser.add_argument('--no-verbose', action='store_true')
	parser.add_argument('--eval_every', type=int, default=5)

	args = parser.parse_args()	

	seed = 42
	print('SEED:', seed)
	cudnn.benchmark = True
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		cudnn.deterministic = True

	# loading data
	args.noise_rate = [float(r) for r in args.noise_rate.split(',')]
	args.na = int(len(args.noise_rate) / 2)
	target = args.dataset
	print('Dataset: {}, lr: {}, noise_rate: {}'.format(target, args.lr, args.noise_rate))
	file_name = 'data/' + target
	(X_folds, Y_folds, Ycs_folds) = load_data_k_fold_cmll(file_name, num_anno=args.na, noise_rate=args.noise_rate)
	K = len(Y_folds)
	if Ycs_folds[0].ndim == 3:
		for i in range(K):
			Ycs_folds[i] = np.mean(Ycs_folds[i], axis=1)
	
	# defining true transition matrix
	args.nc = Y_folds[0].shape[1]
	true_tm = np.zeros((args.na, args.nc, 2, 2))
	for m in range(args.na):
		for i in range(args.nc):
			true_tm[m, i, 0, 0] = 1 - args.noise_rate[2*m]  # 0->0
			true_tm[m, i, 0, 1] = args.noise_rate[2*m]  # 0->1
			true_tm[m, i, 1, 0] = args.noise_rate[2*m+1]  # 1->0
			true_tm[m, i, 1, 1] = 1 - args.noise_rate[2*m+1]  # 1->1
	print('True transition matrix (Aggregated):', true_tm.mean(axis=0))
	n1 = round(true_tm.mean(axis=0)[0][0][1], 1)
	n2 = round(true_tm.mean(axis=0)[0][1][0], 1)
	print('Scenario:', n1, n2)

	# setting vae params
	args.label_dim = Y_folds[0].shape[1]
	args.feature_dim = X_folds[0].shape[1]
	print(args)

	# training each fold
	for i in range(K):
		
		print('\n\nTesting Fold ', i)
		X_test, Y_test = X_folds[i], Y_folds[i]
		K_train = list(set(range(K)) - set([i]))
		
		print('Training Folds ', K_train)
		X_train = np.concatenate([X_folds[j] for j in K_train], axis=0)
		Y_train = np.concatenate([Y_folds[j] for j in K_train], axis=0)
		Y_cs_train = np.concatenate([Ycs_folds[j] for j in K_train], axis=0)

		print('Training set size (X, Y, Y_cs): ', X_train.shape, Y_train.shape, Y_cs_train.shape)
		data = (X_train, Y_train, Y_cs_train, X_test, Y_test)

		# estimating transition matrix
		hidden_size = [64, 64]
		if args.nc >= 64 and args.nc < 256:
			hidden_size = [256, 256]
		if args.nc >= 256:
			hidden_size = [512, 512]
		model_es = DeepNet(X_test.shape[1], Y_test.shape[1], hidden_size).cuda()
		optimizer_es = torch.optim.SGD(model_es.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		t_m = estimate_noise_rate(model_es, data, optimizer_es, args, true_tm.mean(axis=0))
		print('Estimated transition matrix is:', t_m)

		model = VAE(args).cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
		one_epoch_iter = np.ceil(len(X_train) / args.batch_size)  # decay happens if smaller than lr_decay_times
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, one_epoch_iter * (args.epochs / args.lr_decay_times), args.lr_decay_ratio)
		
		# training model
		run_model(args, data, model, optimizer, t_m, eval_every=args.eval_every, scheduler=scheduler)