from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, 
                               self.size, init_method=init_method_1)
        self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, 
                                   self.size, init_method=init_method_2)
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.denom = torch.sqrt(torch.tensor(self.output_size).float())

    def forward(self, ob_no):
        target = self.f(ob_no)
        pred = self.f_hat(ob_no)
        # assert pred.shape == target.shape
        # print(f'\n\nob_no.shape: {ob_no.shape}\nob_no[:5, :]: {ob_no[:5, :]}')
        # print(f'pred.shape: {pred.shape}\npred[:5, :]: {pred[:5, :]}')
        # print(f'target.shape: {target.shape}\ntarget[:5, :]: {target[:5, :]}')
        error_loss = torch.norm(pred - target.detach(), dim=1) / self.denom
        # assert error_loss.shape == pred.shape[:1 ]
        # print(f'error.shape: {error_loss.shape}\nerror[:5]: {error_loss[:5]}')
        return error_loss

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error_loss = self(ob_no)
        return ptu.to_numpy(error_loss)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        ob_no = ptu.from_numpy(ob_no)
        error_loss = self(ob_no).mean()
        self.optimizer.zero_grad()
        error_loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        error_loss = ptu.to_numpy(error_loss)
        return {'Training Loss': error_loss.item()}
