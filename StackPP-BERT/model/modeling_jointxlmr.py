import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier
from torch.autograd import Variable

from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

import math

class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        # seq_lens.cpu(): https://github.com/pytorch/pytorch/issues/43227
        packed_text = pack_padded_sequence(dropout_text, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(   self, input_dim, hidden_dim, output_dim, dropout_rate, \
                    embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.rand(1, self.__embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )
        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        flat_x = torch.cat(
            [attention_x[i][:seq_lens[i], :] for
             i in range(0, len(seq_lens))], dim=0
        )
        return flat_x

class StackPropagationXLMR(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(StackPropagationXLMR, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = XLMRobertaModel(config=config)  # Load pretrained bert

        # for param in self.bert.parameters():
        #     param.requires_grad = False


        # Initialize an LSTM Encoder object.
        self.encoder = LSTMEncoder(
            self.args.word_embedding_dim,
            self.args.encoder_hidden_dim,
            self.args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.attention = SelfAttention(
            self.args.word_embedding_dim,
            self.args.attention_hidden_dim,
            self.args.attention_output_dim,
            self.args.dropout_rate
        )

        # Initialize an Decoder object for intent.
        self.intent_decoder = LSTMDecoder(
            self.args.encoder_hidden_dim + self.args.attention_output_dim,
            self.args.intent_decoder_hidden_dim,
            self.num_intent_labels, self.args.dropout_rate,
            embedding_dim=self.args.intent_embedding_dim
        )
        # Initialize an Decoder object for slot.
        self.slot_decoder = LSTMDecoder(
            self.args.encoder_hidden_dim + self.args.attention_output_dim,
            self.args.slot_decoder_hidden_dim,
            self.num_slot_labels, self.args.dropout_rate,
            embedding_dim=self.args.slot_embedding_dim,
            extra_dim=self.num_intent_labels
        )

        # One-hot encoding for augment data feed. 
        self.intent_embedding = nn.Embedding(
            self.num_intent_labels, self.num_intent_labels
        )
        self.intent_embedding.weight.data = torch.eye(self.num_intent_labels)
        self.intent_embedding.weight.requires_grad = False

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids,
                teacher_forcing=True, is_test=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        seq_lens = torch.sum(attention_mask, dim=1)
        # sequence_output: (batch_size x max_seq_len x word_hidden_size)
        lstm_hiddens = self.encoder(sequence_output, seq_lens)
        # lstm_hiddens: (batch_size*max_seq_len x hidden size)
        attention_hiddens = self.attention(sequence_output, seq_lens)
        # hiddens: (batch_size*max_seq_len x 2*hidden size)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        if not is_test:
            forced_intent = []
            forced_slot = []
            for intent, seq_len in zip(intent_label_ids, seq_lens):
                forced_intent.extend([intent]*seq_len)

            for slot, seq_len in zip(slot_labels_ids, seq_lens):
                forced_slot.extend(slot[:seq_len])

            # forced_intent: (batch_size*total_word_num x 1)
            # forced_slot: (batch_size*total_word_num x 1)
            forced_intent = Variable(torch.LongTensor(forced_intent))
            forced_slot = Variable(torch.LongTensor(forced_slot))

            if torch.cuda.is_available():
                forced_intent = forced_intent.cuda()
                forced_slot = forced_slot.cuda()
        if teacher_forcing:
            # pred_intent: (batch_size x 1)
            pred_intent = self.intent_decoder(
                hiddens, seq_lens,
                forced_input=forced_intent # teacher forcing :D
            )

            if not self.args.differentiable:
                _, idx_intent = pred_intent.topk(1, dim=-1)
                feed_intent = self.intent_embedding(idx_intent.squeeze(1))
            else:
                feed_intent = pred_intent

            # pred_slot: (batch_size*total_word_len x num_slot_label)
            pred_slot = self.slot_decoder(
                hiddens, seq_lens,
                forced_input=forced_slot,
                extra_input=feed_intent
            )
        else:
            pred_intent = self.intent_decoder(
                hiddens, seq_lens,
            )

            if not self.args.differentiable:
                _, idx_intent = pred_intent.topk(1, dim=-1)
                feed_intent = self.intent_embedding(idx_intent.squeeze(1))
            else:
                feed_intent = pred_intent

            # pred_slot: (batch_size*total_word_len x num_slot_label)
            pred_slot = self.slot_decoder(
                hiddens, seq_lens,
                extra_input=feed_intent
            )


        # intent of each tokens
        # intent_logits = F.log_softmax(pred_intent, dim=1)
        # slot_logits = F.log_softmax(pred_slot, dim=1)

        intent_logits = pred_intent
        slot_logits = pred_slot

        # intent_logits = self.intent_classifier(pooled_output)
        # slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), forced_intent.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(  intent_logits.view(-1, self.num_intent_labels), \
                                                forced_intent.view(-1)
                                                )
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                # if attention_mask is not None:
                #     active_loss = attention_mask.view(-1) == 1
                #     active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                #     active_labels = slot_labels_ids.view(-1)[active_loss]
                #     slot_loss = slot_loss_fct(active_logits, active_labels)
                # else:
                slot_loss = slot_loss_fct(  slot_logits.view(-1, self.num_slot_labels), \
                                            forced_slot.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        # print(intent_loss, slot_loss)
        # print(slot_logits.view(-1, self.num_slot_labels), forced_slot.view(-1))

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
