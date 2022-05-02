import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from .modules import IntentClassifier, SlotClassifier

def get_optim(model, args):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    return optimizer


class JointXLMR(nn.Module):

    def __init__(self, pretrained, num_intent, num_slot, dropout) -> None:
        super().__init__()
        config = XLMRobertaConfig.from_pretrained(pretrained)
        self.xlmr = XLMRobertaModel.from_pretrained(pretrained)
        self.intent_dec = IntentClassifier(config.hidden_size, num_intent, dropout_rate = dropout)
        self.slot_dec = SlotClassifier(config.hidden_size, num_slot, dropout_rate = dropout)
        self.crf = CRF(num_slot, batch_first = True)
        self.intent_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, att_mask):
        trf_output = self.xlmr(input_ids, attention_mask = att_mask)
        embedding, pooler = trf_output.last_hidden_state, trf_output.pooler_output

        intent_logits = self.intent_dec(pooler)
        slot_logits = self.slot_dec(embedding)
        return intent_logits, slot_logits

    def get_loss(self, intent_logits, slot_logits, intent_labels, slot_labels, mask, intent_coeff = 0.5):
        intent_loss = self.intent_loss(intent_logits, intent_labels)
        slot_loss = - self.crf(slot_logits, slot_labels, mask.byte(), reduction = 'mean')
        total_loss = intent_loss * intent_coeff + (1 - intent_coeff) * slot_loss
        return total_loss, intent_loss, slot_loss