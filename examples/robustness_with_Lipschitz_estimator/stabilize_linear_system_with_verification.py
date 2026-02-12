# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase


from eclipse_nn.LipConstEstimator import LipConstEstimator

import random
import numpy as np

# Python built-in random
random.seed(42)

# NumPy
np.random.seed(42)

# PyTorch (CPU and GPU)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)   # if using multi-GPU

# Make cuDNN deterministic (slower, but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# Double integrator parameters
nx = 2
nu = 1
A = torch.tensor([[1.2, 1.0],
                  [0.0, 1.0]])
B = torch.tensor([[1.0],
                  [0.5]])

# linear state space model
xnext = lambda x, u: x @ A.T + u @ B.T    
double_integrator = Node(xnext, ['X', 'U'], ['X'], name='integrator')

# We will define the node for the neural control policy in the Optimization Problem Section for different training and wrap the closed-loop system node.

# Training dataset generation
train_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, nx)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

# neural control policy
mlp = blocks.MLP(nx, nu, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])
policy = Node(mlp, ['X'], ['U'], name='policy')

# closed loop system definition
cl_system = System([policy, double_integrator])
# cl_system.show()

# Define optimization problem
u = variable('U')
x = variable('X')
action_loss = 0.0001 * (u == 0.)^2  # control penalty
regulation_loss = 10. * (x == 0.)^2  # target position
loss = PenaltyLoss([action_loss, regulation_loss], [])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)


trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=50,
)

# Train model with prediction horizon of 2
cl_system.nsteps = 2
best_model = trainer.train()


# neural control policy
mlp_reg = blocks.MLP(nx, nu, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])
policy_reg = Node(mlp_reg, ['X'], ['U'], name='policy')




def _jac_fro_node(X):
    # X comes in as [B, nx].
    with torch.enable_grad():
        X = X.detach().requires_grad_(True)          # leaf with grad
        U = mlp_reg(X)                                # [B, nu]
        if U.dim() == 1:
            U = U.unsqueeze(-1)

        sqsum = 0.0
        for j in range(U.shape[1]):
            g = torch.autograd.grad(
                U[:, j].sum(), X,
                create_graph=True, retain_graph=True
            )[0]                                      # [B, nx]
            sqsum = sqsum + (g * g).sum(dim=1)        # [B]

        Jfro = torch.sqrt(sqsum + 1e-12).unsqueeze(-1)  # [B, 1]
    return Jfro

jac_fro = Node(_jac_fro_node, ['X'], ['Jfro'], name='jac_fro')

# closed loop system definition (add the jac_fro node)
cl_system_reg = System([policy_reg, double_integrator, jac_fro])


# variables
u = variable('U')
x = variable('X')
j = variable('Jfro')   # from the Jacobian-Fro node

action_loss     =  0.0001 * (u == 0.)^2       
regulation_loss = 10. * (x == 0.)^2     
jac_loss    = 1 * (j == 0.)^2 

loss_reg = PenaltyLoss([action_loss, regulation_loss,jac_loss], [])
problem_reg = Problem([cl_system_reg], loss_reg)

optimizer = torch.optim.AdamW(policy_reg.parameters(), lr=0.003)

trainer_reg = Trainer(
    problem_reg,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=50,
)


# Train model with prediction horizon of 2
cl_system_reg.nsteps = 2
best_model_reg = trainer_reg.train()



mlp_reg_cl = blocks.MLP(nx, nu, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])
policy_reg_cl = Node(mlp_reg_cl, ['X'], ['U'], name='policy')


def make_jac_cl_node(policy_node):
    """Node: inputs ['X'] (batch,nx) -> outputs ['Jcl'] (batch,1) with ||A + B J_pi||_F^2."""
    policy_module = policy_node.callable  # underlying nn.Module: X -> U

    def _forward(X):                      # <-- NOTE: positional tensor, not dict
        X = X.requires_grad_(True)

        Bsize = X.shape[0]
        vals = []
        for b in range(Bsize):
            xb = X[b]                    # (nx,)

            def f_single(x_single):      # R^nx -> R^nu
                return policy_module(x_single.unsqueeze(0)).squeeze(0)

            J_pi = torch.autograd.functional.jacobian(f_single, xb, create_graph=True)  # (nu,nx)
            J_cl = A + B @ J_pi                                                          # (nx,nx)
            vals.append(torch.linalg.norm(J_cl, ord='fro'))                          # scalar

        Jcl_fro = torch.stack(vals, dim=0).unsqueeze(1)  # (batch,1)
        return Jcl_fro                                   

    return Node(_forward, ['X'], ['Jcl_fro'], name='jac_cl')



jac_cl_node = make_jac_cl_node(policy_reg_cl)

# closed loop system definition (add the jac_fro node)
cl_system_reg_cl = System([policy_reg_cl, double_integrator, jac_cl_node])


# variables
u = variable('U')
x = variable('X')
jcl = variable('Jcl_fro')   # from the Jacobian-Fro node

action_loss     =  0.0001 * (u == 0.)^2       
regulation_loss = 10. * (x == 0.)^2     
jac_cl_loss    = 1 * (jcl == 0.)^2 

loss_reg_cl = PenaltyLoss([action_loss, regulation_loss,jac_cl_loss], [])
problem_reg_cl = Problem([cl_system_reg_cl], loss_reg_cl)

optimizer = torch.optim.AdamW(policy_reg_cl.parameters(), lr=0.003)

trainer_reg_cl = Trainer(
    problem_reg_cl,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=50,
)


# Train model with prediction horizon of 2
cl_system_reg_cl.nsteps = 2
best_model_reg_cl = trainer_reg_cl.train()


# Test best model with prediction horizon of 50
problem.load_state_dict(best_model)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
nsteps = 30
cl_system.nsteps = nsteps
trajectories = cl_system(data)
pltCL(Y=trajectories['X'].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')
pltPhase(X=trajectories['X'].detach().reshape(nsteps+1, 2), figname='phase.png')

# Test best model with prediction horizon of 50
problem_reg.load_state_dict(best_model_reg)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
nsteps = 30
cl_system_reg.nsteps = nsteps
trajectories = cl_system_reg(data)
pltCL(Y=trajectories['X'].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')
pltPhase(X=trajectories['X'].detach().reshape(nsteps+1, 2), figname='phase.png')

# Test best model with prediction horizon of 50
problem_reg_cl.load_state_dict(best_model_reg_cl)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
nsteps = 30
cl_system_reg_cl.nsteps = nsteps
trajectories = cl_system_reg_cl(data)
pltCL(Y=trajectories['X'].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')
pltPhase(X=trajectories['X'].detach().reshape(nsteps+1, 2), figname='phase.png')



# Get NN model parameters
def to_sequential(neuromancer_mlp: nn.Module) -> nn.Sequential:
    """
    Convert neuromancer.blocks.MLP (built with linear_map=torch.nn.Linear)
    into a vanilla torch.nn.Sequential with the same weights/activations.
    """
    # Neuromancer MLP exposes ModuleLists: .linear (Linear layers) and .nonlin (activations)
    src_linear = list(neuromancer_mlp.linear)
    src_nonlin = list(neuromancer_mlp.nonlin)

    layers = []
    for lin, act in zip(src_linear, src_nonlin):
        # Recreate Linear with same shape & bias usage
        new_lin = nn.Linear(lin.in_features, lin.out_features, bias=(lin.bias is not None))
        # Copy parameters
        new_lin.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            new_lin.bias.data.copy_(lin.bias.data)
        layers.append(new_lin)

        # Recreate activation (skip if Identity)
        if not isinstance(act, nn.Identity):
            layers.append(act.__class__())

    return nn.Sequential(*layers)


seq = to_sequential(mlp)
est = LipConstEstimator(model=seq)
# est.model_review()
lip = est.estimate(method='ECLipsE_Fast')
print(f"The Lipschitz estimate for controller from standard training is {lip}")


seq_reg = to_sequential(mlp_reg)
est_reg = LipConstEstimator(model=seq_reg)
lip_reg = est_reg.estimate(method='ECLipsE_Fast')
print(f"The Lipschitz estimate for controller trained with regularization on the Jacobian of policy u=NN(x)  is {lip_reg}")


seq_reg_cl = to_sequential(mlp_reg_cl)
est_reg_cl = LipConstEstimator(model=seq_reg_cl)
lip_reg_cl = est_reg_cl.estimate(method='ECLipsE_Fast')
print(f"The Lipschitz estimate for controller trained with regularization on the Jacobian of policy u=NN(x)  is {lip_reg_cl}")