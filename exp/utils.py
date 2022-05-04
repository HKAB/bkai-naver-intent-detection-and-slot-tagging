import os
import pickle
from dataloader import DataManager
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from models import StackPropagation, StackPropagationAtt, JointXLMR

def get_model(args, num_intent, num_slot):
    if args.model == 'stackprop':
        model = StackPropagation(args.pretrained_model, args.hidden_dim, num_intent, num_slot, args.dropout, max_len = args.max_len)
    elif args.model == 'stackprop_att':
        model = StackPropagationAtt(args.pretrained_model, args.hidden_dim, num_intent, num_slot, args.dropout, max_len = args.max_len)
    elif args.model == 'jointxlmr':
        model = JointXLMR(args.pretrained_model, num_intent, num_slot, args.dropout)

    return model

def get_dataset(args):
    data_file = 'data.pickle'
    if os.path.isfile(data_file):
        with open(data_file, 'rb') as fr:
            dataset = pickle.load(fr)
    else:
        dataset = DataManager(args.data_dir, args.train_folder, args.dev_folder, args.test_folder, max_len=args.max_len, pretrained=args.pretrained_model)
        with open(data_file, 'wb') as fw:
            pickle.dump(dataset, fw)
    return dataset

def create_mask(len_list, max_len = None):
    max_len = len_list[0] if max_len is None else max_len
    mask = torch.zeros(len(len_list), max_len)
    for i, l in enumerate(len_list):
        mask[i, :l] = 1
    return mask


def get_intent_acc(intent_labels, intent_pred):
    # intent_pred = intent_logits.argmax(dim = -1)
    acc = (intent_labels == intent_pred).float().mean()
    return acc, intent_pred

def get_slot_metrics(slot_labels, slot_pred):
    assert len(slot_labels) == len(slot_pred)
    return {
        "slot_precision": precision_score(slot_labels, slot_pred),
        "slot_recall": recall_score(slot_labels, slot_pred),
        "slot_f1": f1_score(slot_labels, slot_pred)
    }

def get_sent_acc(intent_labels, intent_pred, slot_labels, slot_pred):
    intent_correct = (intent_labels == intent_pred)
    slot_correct = torch.tensor([sl == sp for sl, sp in zip(slot_labels, slot_pred)])
    assert len(intent_correct) == len(slot_correct)
    sent_acc = (intent_correct * slot_correct.to(intent_correct)).float().mean()
    # print(intent_correct * slot_correct.cuda())
    return sent_acc