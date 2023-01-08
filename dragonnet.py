from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from sgld import SGLD


def feed_forward_network(nlayers, din, dbody, dout, act_fun, act_output=False):
    """Helper function to make a simple feed forward network"""
    layers = []
    for n in range(nlayers):
        is_last_layer = (n == nlayers - 1)
        is_first_layer = (n == 0)
        act_n = act_fun() if (not is_last_layer or act_output) else nn.Identity()
        din_n = din if is_first_layer else dbody
        dout_n = dbody if not is_last_layer else dout
        layers.extend([nn.Linear(din_n, dout_n), act_n])
    return nn.Sequential(*layers)



class DragonNet(pl.LightningModule):
    def __init__(
        self, din, dbody=200, dhead=100, depth=3, sgld=False, l2=1e-2, num_pseudo_batches=None, burnin=0.5, metrics_buffer_size=100, lr=1e-3
    ):
        super().__init__()
        self.sgld = sgld
        self.l2 = l2
        self.lr = lr

        # the following properties are only used in the Bayesian version with SGLD
        self.num_pseudo_batches = num_pseudo_batches
        self.burnin = burnin  # when using SGLD, burnin is the number of iterations without noise
        self.metrics_buffer_size = metrics_buffer_size
        self.buffer = dict()

        # representation layers
        self.representation = feed_forward_network(depth, din, dbody, dbody, nn.ELU, act_output=True)
        
        # head for the treated and untreated
        self.h0 = feed_forward_network(depth, dbody, dhead, 1, nn.ELU) 
        self.h1 = feed_forward_network(depth, dbody, dhead, 1, nn.ELU)
        
        # propensity score for logistic regression on representation
        self.ht = nn.Linear(dbody, 1)
        
        # epsilon for tmle regularization
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # initialize neural network weights to match dragonnet paper
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.05)

    def forward(self, X, A):
        Z = self.representation(X)
        tlogits = self.ht(Z).squeeze(-1)
        tprob = (torch.sigmoid(tlogits) + 0.01) / 1.02
        Yhat_1 = self.h1(Z).squeeze(-1)
        Yhat_0 = self.h0(Z).squeeze(-1)
        Ypert_1 =  Yhat_1 + self.epsilon / tprob
        Ypert_0 =  Yhat_0 - self.epsilon / (1 - tprob)
        Ypred = A * Yhat_1 + (1 - A) * Yhat_0
        Ypert = Ypred + self.epsilon * (A / tprob - (1 - A) / (1 - tprob))
        return tlogits, Ypert_0, Ypert_1, Ypred, Ypert

    def configure_optimizers(self):
        if self.sgld:
            burnin_e = int(self.burnin / self.num_pseudo_batches)
            num_pseudo_batches = 1  # self.num_pseudo_batches
            opt = SGLD(self.parameters(), self.lr, num_burn_in_steps=self.burnin, num_pseudo_batches=num_pseudo_batches, weight_decay=self.l2, precondition=False)
            sched = optim.lr_scheduler.LinearLR(opt, total_iters=burnin_e, start_factor=1.0, end_factor=1e-2)
            # return opt
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": 'train_loss'}}
        else:
            opt = optim.SGD(self.parameters(), self.lr, momentum=0.9, nesterov=True, weight_decay=self.l2)
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, cooldown=0, min_lr=0.0, patience=10, factor=0.5, verbose=True)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": 'train_loss'}}

    def training_step(self, train_batch, batch_idx):
        # evaluate network prediction on minibatch
        X, A, Y = train_batch
        tlogits, Ypert_0, Ypert_1, Ypred, Ypert = self(X, A)

        # compute multiple parts of the loss function
        yloss = F.mse_loss(Y, Ypred, reduction='mean')  # standard mse loss for the outcome 
        tloss = F.binary_cross_entropy_with_logits(tlogits, A, reduction='mean')  # logistic loss or treatment
        
        # now compute tmle regularization loss of dragonnet
        tregloss = F.mse_loss(Y, Ypert, reduction='mean')  # tmle loss
        
        # total loss and causal effect (ate) estimate
        loss = yloss + tloss + tregloss
        ate_estimate = (Ypert_1 - Ypert_0).mean()

        # log metrics
        self.log('train_loss', float(loss), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('mse_loss', float(yloss), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('treatment_loss', float(tloss), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('treg_loss', float(tregloss), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('ate_estimate', float(ate_estimate), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        # self.log('lr', float(self.lr_schedulers().get_last_lr()[0]), prog_bar=True, on_epoch=True, on_step=False)

        return loss  # returns the loss to be optimized with SGD/SGLD

    def on_train_epoch_end(self) -> None:
        for k, v in self.trainer.logged_metrics.items():
            if k not in self.buffer:
                self.buffer[k] = deque(maxlen=self.metrics_buffer_size)
            self.buffer[k].append(v.item())