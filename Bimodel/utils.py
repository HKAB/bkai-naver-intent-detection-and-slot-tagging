import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from dataloader import DataManager

def get_dataset(args):
    pretrained = args.pretrained_model if args.pretrained else None
    dataset = DataManager(args.data_dir, args.train_folder, args.dev_folder, args.test_folder, max_len=args.max_len, pretrained=pretrained)
    return dataset

def create_mask(len_list, max_len = None):
    max_len = len_list[0] if max_len is None else max_len
    mask = torch.zeros(len(len_list), max_len)
    for i, l in enumerate(len_list):
        mask[i, :l] = 1
    return mask

def get_intent_acc(intent_labels, intent_logits):
    intent_pred = intent_logits.argmax(dim = 1)
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
    slot_acc = slot_correct.float().mean()
    return sent_acc, slot_acc