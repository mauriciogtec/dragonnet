# -----
# Copy of https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html#SGLD
# modified by Mauricio Tec 9/14/2022
# modification: add weight_decay
# -----

import torch
from torch.optim import Optimizer

# Pytorch Port of a previous tensorflow implementation in `tensorflow_probability`:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics.md

class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in each dimension
        according to RMSProp.
    """
    def __init__(
        self, params, lr=1e-2, precondition_decay_rate=0.95, num_pseudo_batches=1, num_burn_in_steps=3000, diagonal_bias=1e-8, weight_decay=0.0, precondition=True
    ) -> None:
        """ Set up a SGLD Optimizer.
            Parameters
            ----------
            params : iterable
                Parameters serving as optimization variable.
            lr : float, optional
                Base learning rate for this optimizer.
                Must be tuned to the specific function being minimized.
                Default: `1e-2`.
            precondition_decay_rate : float, optional
                Exponential decay rate of the rescaling of the preconditioner (RMSprop).
                Should be smaller than but nearly `1` to approximate sampling from the posterior.
                Default: `0.95`
            preconditioning: bool, optional
                When False it does not scale the gradients.
            num_pseudo_batches : int, optional
                Effective number of minibatches in the data set.
                Trades off noise and prior with the SGD likelihood term.
                Note: Assumes loss is taken as mean over a minibatch.
                Otherwise, if the sum was taken, divide this number by the batch size.
                Default: `1`.
            num_burn_in_steps : int, optional
                Number of iterations to collect gradient statistics to update the
                preconditioner before starting to draw noisy samples.
                Default: `3000`.
            diagonal_bias : float, optional
                Term added to the diagonal of the preconditioner to prevent it from
                degenerating.
                Default: `1e-8`.
            weight_decay : float, optional
                Standard weight decay (providing l2 regularization) added to gradients.
                Default: `1e-2`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=diagonal_bias,
            weight_decay=weight_decay,
            precondition=precondition
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data
                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.tensor(0.0).to(parameter.device)

                if group['precondition']:
                    preconditioner = (
                        1. / torch.sqrt(momentum + group["diagonal_bias"])
                    )
                else:
                    preconditioner = torch.ones_like(momentum)

                scaled_grad = (
                    0.5 * gradient * num_pseudo_batches +
                    torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma / preconditioner
                )
                if group['weight_decay'] != 0:
                    scaled_grad.add_(parameter.data, alpha=group['weight_decay'])

                parameter.data.add_(-lr * scaled_grad)

        return loss