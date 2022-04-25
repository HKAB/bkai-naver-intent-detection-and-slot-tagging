import torch
import torch.nn as nn
from model.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointSlotRefine(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointSlotRefine, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.slot_label_lst = slot_label_lst

        self.o_tag_idx = slot_label_lst.index("O")

        self.slot_embedding = nn.Embedding(len(slot_label_lst), config.hidden_size)

        self.bert = RobertaModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):



        # first pass, all embedding are "O"
        first_phase_embedding = torch.ones_like(input_ids) * self.o_tag_idx
        first_phase_embedding = self.slot_embedding(first_phase_embedding)

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            additional_embedding=first_phase_embedding)  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        first_pass_intent_logits = self.intent_classifier(pooled_output)
        first_pass_slot_logits = self.slot_classifier(sequence_output)

        second_phase_embedding = torch.argmax(first_pass_slot_logits, dim=2)
        for i in range(second_phase_embedding.shape[0]):
            for j in range(second_phase_embedding.shape[1]):
                if ("B-" not in self.slot_label_lst[second_phase_embedding[i][j]]):
                    second_phase_embedding[i][j] = self.o_tag_idx

        second_phase_embedding = self.slot_embedding(second_phase_embedding)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    additional_embedding=second_phase_embedding)  # sequence_output, pooled_output, (hidden_states), (attentions)        

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        second_pass_intent_logits = self.intent_classifier(pooled_output)
        second_pass_slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(first_pass_intent_logits.view(-1), intent_label_ids.view(-1))
                intent_loss += intent_loss_fct(second_pass_intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(first_pass_intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
                intent_loss += intent_loss_fct(second_pass_intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(first_pass_slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss += self.crf(second_pass_slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    first_pass_active_logits = first_pass_slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    second_active_logits = second_pass_slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(first_pass_active_logits, active_labels)
                    slot_loss += slot_loss_fct(second_active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(first_pass_slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
                    slot_loss += slot_loss_fct(second_pass_slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((second_pass_intent_logits, second_pass_slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
