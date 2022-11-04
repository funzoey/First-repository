import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# from metron.builders import required_field, optional_field

def backward2forward(labels, eos):
    # labels: [bs, tgt_len]
    bsz, tgt_len = labels.shape
    _, eos_position = torch.nonzero(labels == eos, as_tuple=True)
    eos_position = eos_position.cpu()
    increase = torch.arange(tgt_len).view(1, tgt_len).repeat(bsz, 1) # 0 to tgt_len for each row
    decrease = eos_position.view(bsz, 1).expand(-1, tgt_len) - increase - 1
    mask = increase < eos_position.view(bsz, 1)
    increase[mask] = decrease[mask]
    return (increase + torch.arange(bsz).view(bsz, 1) * tgt_len).flatten(), mask

class KLLossCriterion(_Loss):
    def __init__(self, config, eos, criterion):
        super().__init__()
        self.eos = eos
        self.ensemble_loss = criterion
        # self.ensemble_loss_weight = required_field(config, 'ensemble_loss_weight')

        self.kl_loss = torch.nn.KLDivLoss(reduction='none')
        # self.logit_kl_temperature = optional_field(config, 'logit_kl_temperature', 1.0)
        # self.logit_kl_weight = required_field(config, 'logit_kl_weight') * self.logit_kl_temperature * self.logit_kl_temperature

    def forward(self, outputs, reversed_outputs, labels):
        '''
        outputs, reversed_outputs: [bsz, tgt_len, V], logits
        labels: [bsz, tgt_len]
        '''
        bsz, tgt_len = labels.shape
        backward2forward_index, mask = backward2forward(labels, self.eos) # [bsz * tgt_len], [bsz, tgt_len]
        backward2forward_index = backward2forward_index.to(outputs.device)
        mask = torch.logical_not(mask).to(outputs.device)

        outputs_r2l_logit = torch.index_select(
            reversed_outputs.view(bsz * tgt_len, -1),
            0, backward2forward_index
        ).view_as(outputs) # [bsz, tgt_len, V]

        # ensemble loss
        ensemble_logit = (outputs + outputs_r2l_logit) * 0.5
        ensemble_loss, _, _ = self.ensemble_loss(ensemble_logit, labels, reduce=False)
        ensemble_loss = ensemble_loss.view(bsz, tgt_len)

        # distillation loss
        kl_target = F.softmax(ensemble_logit.detach() / self.logit_kl_temperature, dim=-1)

        # -target*log(input) # [bsz, tgt_len]
        kl_l2r = self.kl_loss(
            F.log_softmax(outputs / self.logit_kl_temperature, dim=-1),
            kl_target,
        ).sum(dim=-1)
        kl_r2l = self.kl_loss(
            F.log_softmax(outputs_r2l_logit / self.logit_kl_temperature, dim=-1),
            kl_target,
        ).sum(dim=-1)

        kl_l2r.masked_fill_(mask.cuda(), 0)
        kl_r2l.masked_fill_(mask.cuda(), 0)

        return ensemble_loss *  self.ensemble_loss_weight + (kl_l2r + kl_r2l) * self.logit_kl_weight
