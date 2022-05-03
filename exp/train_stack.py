import argparse
from tqdm import tqdm
import json
import logging
import pickle
import os

from dataloader import *
from utils import *
from models import StackPropagation

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save_dir, 'log.txt'),
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)
            
    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
    # device = torch.device('cpu')

    dataset = get_dataset(args)
    
    train_data = dataset.get_data('train')
    num_word = len(dataset.word_dict.keys())
    num_slot = len(dataset.slot_label)
    num_intent = len(dataset.intent_label)
    loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, num_workers = 8, shuffle = True, pin_memory = True)

    dev_data = dataset.get_data('dev')
    dev_loader = DataLoader(dev_data, batch_size = args.dev_batch_size, collate_fn = dev_data.collate_fn, num_workers = 8, shuffle = False, pin_memory = True)

    model = StackPropagation(args.pretrained_model, args.hidden_dim, num_intent, num_slot, args.dropout, max_len = args.max_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    best_acc = 0
    best_ckpt = 0
    iterator = tqdm(range(args.num_epoch))
    for i in iterator:
        model.train()
        for text, att_mask, slots, intents, len_list, perm_idx in loader:
            text = text.to(device)
            slots = slots.to(device)
            intents = intents.to(device)
            att_mask = att_mask.to(device)

            # mask = create_mask(len_list, args.max_len).to(device)

            optimizer.zero_grad()
            intent_logits, slot_logits = model(text, att_mask, len_list)

            loss, intent_loss, slot_loss = model.get_loss(intent_logits, slot_logits, intents, slots, att_mask, intent_coeff = args.intent_coeff)
            loss.backward()
            optimizer.step()
        
        intent_acc, slot_metrics, sent_acc = evaluate(model, dev_loader, device, dataset.slot_label)
        slot_f1 = slot_metrics['slot_f1']
        slot_pre = slot_metrics['slot_precision']
        slot_recall = slot_metrics['slot_recall']
        if sent_acc >= best_acc:
            best_acc = sent_acc
            best_ckpt = i
            torch.save(model.state_dict(), f'{args.save_dir}/model.pth')
            
        logging.info(f'Epoch: {i}, Total loss: {loss:.2f}, Intent loss: {intent_loss.item():.2f}, Slot loss, {slot_loss.item():.2f}, Intent acc: {intent_acc:.2f}, Slot F1: {slot_f1:.2f}, Slot pre: {slot_pre:.2f}, Slot recall: {slot_recall:.2f}, Sent acc: {sent_acc:.2f}')
        iterator.set_description(f'Epoch: {i}, Total loss: {loss:.2f}, Intent loss: {intent_loss.item():.2f}, Slot loss, {slot_loss.item():.2f}, Intent acc: {intent_acc:.2f}, Slot F1: {slot_f1:.2f}, Sent acc: {sent_acc:.2f}')
    logging.info(f'Best checkpoint: {best_ckpt}')


def get_prediction(model, dev_loader, device, slot_list, is_test = False):
    model.eval()
    slot_pred = []
    intent_pred = []
    intent_label = []
    slot_label = []
    with torch.no_grad():
        for batch in dev_loader:
            if not is_test:
                text, att_mask, slots, intents, len_list, perm_idx = batch
                intents = intents.to(device)
            else:
                text, att_mask, slots, len_list, perm_idx = batch
            text = text.to(device)
            slots = slots.to(device)
            att_mask = att_mask.to(device)

            intent_out, slot_out = model.predict(text, att_mask, slots, len_list, device, perm_idx = perm_idx, intents = intents)
            # intent_out, slot_out = model(text, att_mask, len_list)
            # intent_out = model.intent_dec.crf.decode(intent_out)
            # seq_mask = slots >= 0
            # intent_out = model.get_intent(intent_out, seq_mask).to(device)
            # slot_out = model.slot_dec.crf.decode(slot_out)

            if not is_test:
                intent_label.append(intents)
            intent_pred.append(intent_out)

            for sp, sl in zip(slot_out, slots):
                assert len(sp) == len(sl)
                slot_label.append([slot_list[tgt_slot] for tgt_slot in sl if tgt_slot >= 0])
                slot_pred.append([slot_list[out_slot] for out_slot, tgt_slot in zip(sp, sl) if tgt_slot >= 0])

    intent_pred = torch.cat(intent_pred)
    if is_test:
        return intent_pred, slot_pred
    else:
        intent_label = torch.cat(intent_label)
        return intent_label, intent_pred, slot_label, slot_pred

def evaluate(model, dev_loader, device, slot_list):
    intent_label, intent_pred, slot_label, slot_pred = get_prediction(model, dev_loader, device, slot_list)
    intent_acc, intent_pred = get_intent_acc(intent_label, intent_pred)
    slot_metrics = get_slot_metrics(slot_label, slot_pred)
    sent_acc = get_sent_acc(intent_label, intent_pred, slot_label, slot_pred)
    return intent_acc, slot_metrics, sent_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--train-folder', type=str, default='training_data')
    parser.add_argument('--dev-folder', type=str, default='dev_data')
    parser.add_argument('--test-folder', type=str, default='public_test_data')
    parser.add_argument('--save-dir', type=str, default='save/stackprop')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained-model', type=str, default='xlm-roberta-base')
    parser.add_argument('--emb-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type = int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--intent-coeff', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dev-batch-size', type=int, default=32)
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--num-epoch', type=int, default=500)
    
    args = parser.parse_args()
    train(args)