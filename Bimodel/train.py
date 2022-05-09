import argparse
from dataloader import *
from utils import *
from models import WordEmbedding, SlotModel, IntentModel
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import json
import logging

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

    emb_dim = args.emb_dim

    if args.pretrained:
        pretrained_embedding = AutoModel.from_pretrained(args.pretrained_model)#.requires_grad_(False)
        emb_dim = AutoConfig.from_pretrained(args.pretrained_model).hidden_size

    embedding = WordEmbedding(num_word, args.emb_dim) if not args.pretrained else pretrained_embedding
    slot_model = SlotModel(embedding, emb_dim, args.hidden_dim, num_slot, \
        args.dropout, args.max_len, use_pretrained = args.pretrained).to(device)

    if args.pretrained:
        pretrained_embedding = AutoModel.from_pretrained(args.pretrained_model)#.requires_grad_(False)
        emb_dim = AutoConfig.from_pretrained(args.pretrained_model).hidden_size
    embedding = WordEmbedding(num_word, args.emb_dim) if not args.pretrained else pretrained_embedding
    intent_model = IntentModel(embedding, emb_dim, args.hidden_dim, num_intent, \
        args.dropout, args.max_len, use_pretrained = args.pretrained).to(device)

    intent_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, intent_model.parameters()), lr = args.lr)
    slot_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, slot_model.parameters()), lr = args.lr)

    best_acc = 0
    iterator = tqdm(range(args.num_epoch))
    for i in iterator:
        slot_model.train()
        intent_model.train()
        for text, att_mask, slots, intents, len_list, perm_idx in loader:
            text = text.to(device)
            slots = slots.to(device)
            intents = intents.to(device)
            att_mask = att_mask.to(device)

            # mask = create_mask(len_list, args.max_len).to(device)

            intent_optim.zero_grad()
            slot_optim.zero_grad()
            intent_feat, pooler = intent_model.encode(text, len_list, att_mask = att_mask)
            intent_share = intent_feat.clone().detach()
            slot_feat = slot_model.encode(text, len_list, att_mask = att_mask)
            slot_share = slot_feat.clone().detach()
            # print(intent_share, slot_share, intent_feat, slot_feat)
            
            intent_out = intent_model.decode(intent_feat, slot_share, len_list, pooler = None)
            intent_loss = intent_model.get_loss(intent_out, intents)
            torch.nn.utils.clip_grad_norm_(intent_model.parameters(), 5.0)
            intent_loss.backward()
            intent_optim.step()
            
            
            slot_out = slot_model.decode(slot_feat, intent_share, len_list)
            slot_loss = slot_model.get_loss(slot_out, slots, att_mask)
            torch.nn.utils.clip_grad_norm_(slot_model.parameters(), 5.0)
            slot_loss.backward()
            slot_optim.step()
        
        intent_acc, slot_metrics, sent_acc, slot_acc = evaluate(slot_model, intent_model, dev_loader, device, dataset.slot_label)
        slot_f1 = slot_metrics['slot_f1']
        slot_pre = slot_metrics['slot_precision']
        slot_recall = slot_metrics['slot_recall']
        if sent_acc >= best_acc:
            best_acc = sent_acc
            torch.save(slot_model.state_dict(), f'{args.save_dir}/slot.pth')
            torch.save(intent_model.state_dict(), f'{args.save_dir}/intent.pth')
            
        logging.info(f'Epoch: {i}, Intent loss: {intent_loss.item():.2f}, Slot loss, {slot_loss.item():.2f}, Intent acc: {intent_acc:.2f}, Slot F1: {slot_f1:.2f}, Slot pre: {slot_pre:.2f}, Slot recall: {slot_recall:.2f}, Slot acc: {slot_acc:.2f}, Sent acc: {sent_acc:.2f}')
        iterator.set_description(f'Epoch: {i}, Intent loss: {intent_loss.item():.2f}, Slot loss, {slot_loss.item():.2f}, Intent acc: {intent_acc:.2f}, Slot F1: {slot_f1:.2f}, Slot acc: {slot_acc:.2f}, Sent acc: {sent_acc:.2f}')

def evaluate(slot_model, intent_model, dev_loader, device, slot_list):
    slot_model.eval()
    intent_model.eval()
    slot_pred = []
    intent_logits = []
    intent_label = []
    slot_label = []
    all_len = []
    with torch.no_grad():
        for batch in dev_loader:
            text, att_mask, slots, intents, len_list, perm_idx = batch
            text = text.to(device)
            slots = slots.to(device)
            intents = intents.to(device)
            att_mask = att_mask.to(device)
            all_len.append(len_list)

            intent_feat, pooler = intent_model.encode(text, len_list, att_mask = att_mask)
            intent_share = intent_feat.clone().detach()
            slot_feat = slot_model.encode(text, len_list, att_mask = att_mask)
            slot_share = slot_feat.clone().detach()
            
            intent_out = intent_model.decode(intent_feat, slot_share, len_list, pooler = None)
            slot_out = slot_model.decode(slot_feat, intent_share, len_list)
            slot_out = slot_model.crf.decode(slot_out)

            intent_label.append(intents)
            intent_logits.append(intent_out)

            for sp, sl in zip(slot_out, slots):
                assert len(sp) == len(sl)
                # print(sp, sl)
                if len([slot_list[out_slot] for out_slot, tgt_slot in zip(sp, sl) if tgt_slot >= 0]) == 0:
                    print(sp, sl)
                slot_label.append([slot_list[tgt_slot] for tgt_slot in sl if tgt_slot >= 0])
                slot_pred.append([slot_list[out_slot] for out_slot, tgt_slot in zip(sp, sl) if tgt_slot >= 0])
            # slot_label.append(slots)
            # slot_pred.extend(slot_out)

    # print(slot_label, slot_pred)
    intent_label = torch.cat(intent_label)
    intent_logits = torch.cat(intent_logits)

    intent_acc, intent_pred = get_intent_acc(intent_label, intent_logits)
    slot_metrics = get_slot_metrics(slot_label, slot_pred)
    sent_acc, slot_acc = get_sent_acc(intent_label, intent_pred, slot_label, slot_pred)
    return intent_acc, slot_metrics, sent_acc, slot_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--train-folder', type=str, default='training_data')
    parser.add_argument('--dev-folder', type=str, default='dev_data')
    parser.add_argument('--test-folder', type=str, default='public_test_data')
    parser.add_argument('--save-dir', type=str, default='save/default')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained-model', type=str, default='xlm-roberta-base')
    parser.add_argument('--emb-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type = int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dev-batch-size', type=int, default=32)
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--num-epoch', type=int, default=500)
    
    args = parser.parse_args()
    train(args)