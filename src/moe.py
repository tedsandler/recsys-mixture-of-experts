
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader



class MoE(nn.Module):
    def __init__(self, gating_net : nn.Module, experts : list[nn.Module]):
        """
        :gating_net: gating network
        :experts: list of expert networks
        """
        super().__init__()
        self.gating_net = gating_net
        self.experts = self.linears = nn.ModuleList(experts)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        probs = self.gating_net(x)
        logits = [ expert(x) for expert in self.experts ]
        return {
            "probs" : probs,
            "logits" : logits,
        }

    @torch.no_grad()
    def predict(self, x: Tensor, predict_rule: str = "argmax", debug=False) -> Tensor:
        self.eval()
        dataloader = DataLoader(x, batch_size=100, shuffle=False)
        yhats = []
        for x_batch in dataloader:
            fwd_out = self.forward(x_batch)
            probs = fwd_out['probs']
            logits = torch.stack(fwd_out['logits'], dim=1)
            if predict_rule == "argmax":
                jj = probs.argmax(dim=1)
                ii = torch.arange(len(logits))
                yhats.append(logits[ii, jj])
            elif predict_rule == "mixture":
                yhats.append(probs * logits).sum(axis=1)
            else:
                raise ValueError(f"unknown {predict_rule=}")
        if debug:
            import pdb; pdb.set_trace()
        return torch.concat(yhats)


class MOELoss(nn.Module):
    def __init__(self, expert_log_loss_fn=None):
        super().__init__()
        self.expert_log_loss_fn = expert_log_loss_fn
    
    def forward(self, preds: dict[str, Tensor], y: Tensor) -> Tensor:
        loss_fn = self.expert_log_loss_fn
        experts_probs = preds['probs']
        experts_logits = preds['logits']
        log_losses = [loss_fn(logits, y) for logits in experts_logits]
        log_losses = torch.concat(log_losses, dim=1)
        p_Y_given_E = torch.exp(-log_losses)
        p_Y_given_X = (p_Y_given_E * experts_probs).sum(dim=1)
        loss = -torch.log(p_Y_given_X).mean()
        return loss
