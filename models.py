from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class DeepNet(nn.Module):
    def __init__(self, dim_x, dim_y, hidden=None):
        super(DeepNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden[1], dim_y)
        )
        # Sigmoid is necessary to get smooth score

    def forward(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y

    def forward_and_get_fea(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y, fea


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        # feature layers
        self.fx1 = nn.Linear(args.feature_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)  # diagonal

        self.fd_x1 = nn.Linear(args.feature_dim + args.latent_dim, 256)
        self.fd_x2 = nn.Linear(256, 512)
        self.feat_mp_mu = nn.Linear(512, args.label_dim)

        # label layers
        self.fe1 = nn.Linear(args.feature_dim + args.label_dim, 512)
        self.fe2 = nn.Linear(512, 256) 
        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)  # diagonal

        self.fd1 = nn.Linear(args.feature_dim + args.latent_dim, 256)  # decoupled
        self.fd2 = nn.Linear(256, 512)  # decoupled
        self.label_mp_mu = nn.Linear(512, args.label_dim)

        # things they share
        self.dropout = nn.Dropout(p=args.keep_prob)
        self.scale_coeff = args.scale_coeff
        self.r_sqrt_sigma = nn.Parameter(torch.from_numpy(
            np.random.uniform(
                -np.sqrt(6.0/(args.label_dim+args.label_dim)), np.sqrt(6.0/(args.label_dim+args.label_dim)), (args.label_dim, args.label_dim)
            )
        ))

    def label_encode(self, x):
        h1 = self.dropout(F.relu(self.fe1(x)))
        h2 = self.dropout(F.relu(self.fe2(h1)))
        mu = self.fe_mu(h2) * self.scale_coeff
        logvar = self.fe_logvar(h2) * self.scale_coeff
        return mu, logvar

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff
        logvar = self.fx_logvar(h3) * self.scale_coeff
        return mu, logvar

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  # sampling
        return mu + eps*std

    def feat_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def label_decode(self, z):
        h3 = F.relu(self.fd1(z))
        h4 = F.relu(self.fd2(h3))
        return self.label_mp_mu(h4)

    def feat_decode(self, z):
        h4 = F.relu(self.fd_x1(z))
        h5 = F.relu(self.fd_x2(h4))
        return self.feat_mp_mu(h5)

    def label_forward(self, x, feat):
        x = torch.cat((feat, x), 1)
        mu, logvar = self.label_encode(x)
        z = self.label_reparameterize(mu, logvar)
        return self.label_decode(torch.cat((feat, z), 1)), mu, logvar, z  # feature stands for input x

    def feat_forward(self, x):
        mu, logvar = self.feat_encode(x)
        z = self.feat_reparameterize(mu, logvar)
        return self.feat_decode(torch.cat((x, z), 1)), mu, logvar, z

    def forward(self, label, feature):
        # out is mp_mu
        label_out, label_mu, label_logvar, label_z = self.label_forward(label, feature)
        feat_out, feat_mu, feat_logvar, feat_z = self.feat_forward(feature)
        return label_out, label_mu, label_logvar, label_z, feat_out, feat_mu, feat_logvar, feat_z


def compute_BCE_loss(E, input_label):
    # compute negative log likelihood (BCE loss) for each sample point
    sample_nll = -(torch.log(E) * input_label + torch.log(1-E) * (1-input_label))
    logprob = -torch.sum(sample_nll, dim=2)

    # the following computation is designed to avoid the float overflow (log_sum_exp trick)
    maxlogprob = torch.max(logprob, dim=0)[0]
    Eprob = torch.mean(torch.exp(logprob - maxlogprob), axis=0)
    nll_loss = torch.mean(-torch.log(Eprob) - maxlogprob)

    return nll_loss

def compute_BCE_loss_T(E, input_label, t_m):
    weights1 = torch.zeros_like(input_label).float().cuda()
    for i_k in range(input_label.shape[1]):
        weights1[:, i_k] = (input_label[:, i_k] - t_m[i_k, 0, 1]) / (1 - t_m[i_k, 0, 1] - t_m[i_k, 1, 0])
    weights1 = torch.clamp(weights1, min=0, max=1)
    
    # compute negative log likelihood (BCE loss) for each sample point
    sample_nll = -(torch.log(E) * weights1 + torch.log(1-E) * (1-weights1))
    logprob = -torch.sum(sample_nll, dim=2)

    # the following computation is designed to avoid the float overflow (log_sum_exp trick)
    maxlogprob = torch.max(logprob, dim=0)[0]
    Eprob = torch.mean(torch.exp(logprob - maxlogprob), axis=0)
    nll_loss = torch.mean(-torch.log(Eprob) - maxlogprob)

    return nll_loss

def compute_loss(input_label, fe_out, fe_mu, fe_logvar, fx_out, fx_mu, fx_logvar, label_z, feat_z, r_sqrt_sigma, args, n_sample, t_m):
    
    # loss 1: KL distillation loss
    kl_loss = torch.mean(0.5 * \
                         torch.sum((fx_logvar-fe_logvar)-1 + \
                                    torch.exp(fe_logvar-fx_logvar) + \
                                    torch.square(fx_mu-fe_mu) / (torch.exp(fx_logvar)+1e-6), \
                                        dim=1))
    
    # loss 2: MSE distillation loss
    distill_loss = F.mse_loss(feat_z, label_z) + F.mse_loss(fx_out, fe_out)
        
    eps1 = torch.tensor([1e-6]).float().cuda()

    n_sample = n_sample
    n_batch = fe_out.shape[0]

    # standard Gaussian samples
    noise = torch.normal(0, 1, size=(n_sample, n_batch, args.label_dim)).cuda()
    B = r_sqrt_sigma.T.float().cuda()
    sample_r = torch.tensordot(noise, B, dims=1) + fe_out  # tensor: n_sample*n_batch*label_dim
    sample_r_x = torch.tensordot(noise, B, dims=1) + fx_out  # tensor: n_sample*n_batch*label_dim
    norm = torch.distributions.normal.Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
    
    # the probabilities w.r.t. every label in each sample from the batch
    # size: n_sample * n_batch * label_dim
    # eps1: to ensure the probability is non-zero
    # ranging (0,1)
    E = norm.cdf(sample_r) * (1-eps1) + eps1*0.5
    E_x = norm.cdf(sample_r_x) * (1-eps1) + eps1*0.5

    # loss 3: unbiased reconstruction loss
    # label branch
    if args.loss_type == 'bce':
        nll_loss = compute_BCE_loss(E, input_label)
    elif args.loss_type == 'unbiased':
        nll_loss = compute_BCE_loss_T(E, input_label, t_m)
    # feature branch
    if args.loss_type == 'bce':
        nll_loss_x = compute_BCE_loss(E_x, input_label)
    elif args.loss_type == 'unbiased':
        nll_loss_x = compute_BCE_loss_T(E_x, input_label, t_m)

    # if in the testing phase, the prediction 
    indiv_prob = torch.mean(E_x, axis=0)

    # total loss
    total_loss = (nll_loss + nll_loss_x) * args.nll_coeff + kl_loss * 1.1 + distill_loss * args.d_coeff

    return total_loss, nll_loss, nll_loss_x, kl_loss, indiv_prob