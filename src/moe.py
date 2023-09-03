
from typing import Union

import torch
from torch import nn, Tensor


class MoE(nn.Module):
    def __init__(self, gating_net : nn.Module, experts : list[nn.Module]):
        """
        :gating_net: gating network
        :experts: list of expert networks
        """
        super().__init__()
        self.gating_net = gating_net
        self.experts = self.linears = nn.ModuleList(experts)

    def forward(self, x: Tensor, predict_rule: str = "argmax") -> Union[Tensor, dict[str, Tensor]]:
        """
        In evaluation mode, this function returns a torch Tensor of
        predicted $\hat{y}$ values.

        In training mode, this function returns a dictionary with fields
            - expert_probs: a tensor of the expert responsibility weights
            - expert_yhats: a tensor of the expert $\hat{y}$ values

        :param x: input tensor
        :param predict_rule: one of {"argmax", "mean"}. If "argmax", then
            the prediction will be generated by the expert deemed to be the
            best by the gating network. If "mean", then the prediction will
            be a weighted average of the experts' predictions where the
            weighting is chosen by the gating network.
        """
        expert_probs = self.gating_net(x)
        expert_yhats = torch.concat([expert(x) for expert in self.experts], dim=1)

        if self.training:
            return {
                "expert_probs" : expert_probs,
                "expert_yhats" : expert_yhats,
            }

        # otherwise we are in predict mode
        if predict_rule == "argmax":
            ii = torch.arange(len(expert_probs))
            jj = expert_probs.argmax(dim=1)
            yhats = expert_yhats[ii,jj]

        elif predict_rule == "mean":
            yhats = (expert_probs * expert_yhats).sum(dim=1)

        else:
            raise ValueError(f"unknown predict rule '{predict_rule}")

        return yhats.view(-1, 1)


class MOELoss(nn.Module):
    def __init__(self, expert_log_loss_fn=None):
        super().__init__()
        self.expert_log_loss_fn = expert_log_loss_fn
    
    def forward(self, preds: dict[str, Tensor], y: Tensor) -> Tensor:
        loss_fn = self.expert_log_loss_fn
        p_E_given_X = preds['expert_probs']
        yhats = preds['expert_yhats']
        log_losses = []
        for m in range(yhats.shape[1]):
            yhat_m = yhats[:,m].view(-1,1)
            log_losses.append(loss_fn(yhat_m, y))
        log_losses = torch.concat(log_losses, dim=1)
        p_Y_given_E = torch.exp(-log_losses)
        p_Y_given_X = (p_Y_given_E * p_E_given_X).sum(dim=1)
        loss = -torch.log(p_Y_given_X).mean()
        return loss



