from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        obs = ptu.from_numpy(obs)
        dist = self.forward(obs)
        if sample:
            action = dist.rsample()
        else:
            action = dist.mean
        action = action.unsqueeze(0)
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        if self.discrete:
            logits = self.logits_na(observation)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd.clip(*self.log_std_bounds))
            return sac_utils.SquashedNormal(loc=mean, scale=std)

    def update(self, obs, critic):
        ac_dist = self.forward(obs)
        ac = ac_dist.rsample()
        Q, _ = critic(obs, ac)
        # Q1, Q2 = critic(obs, ac)
        # Q = torch.minimum(Q1, Q2)
        logprob = ac_dist.log_prob(ac.clip(-1 + 1e-6, 1 - 1e-6)).sum(-1)
        actor_loss = torch.mean(-Q + self.alpha.detach() * logprob)
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        mult = torch.mean(logprob + self.target_entropy)
        alpha_loss = -self.alpha * mult.detach()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss, alpha_loss, self.alpha

