import json
from dataloader import *
from models import *
from utils import get_dataset, get_model
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig

save_dir = 'large'
with open(os.path.join(save_dir, 'config.json'), 'r') as f:
    args = json.load(f)
class HP:
    def __init__(self, **entries):
        self.__dict__.update(entries)
args = HP(**args)
args.gpu = -1
device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
dataset = get_dataset(args)#DataManager(args.data_dir, args.train_folder, args.dev_folder, args.test_folder, max_len=args.max_len, pretrained=args.pretrained_model)
num_word = len(dataset.word_dict.keys())
num_slot = len(dataset.slot_label)
num_intent = len(dataset.intent_label)
test_data = dataset.get_data('test')
test_loader = DataLoader(test_data, batch_size = args.dev_batch_size, collate_fn = test_data.collate_fn, shuffle = False, pin_memory = False)

model = get_model(args, num_intent, num_slot).to(device)
# model = StackPropagationAtt(args.pretrained_model, args.hidden_dim, num_intent, num_slot, args.dropout, max_len = args.max_len).to(device)
# model = JointXLMR(args.pretrained_model, num_intent, num_slot, args.dropout).to(device)
model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth'), map_location = device))

slot_list = dataset.slot_label
intent_list = dataset.intent_label

model.eval()
slot_pred = []
intent_pred = []
with torch.no_grad():
    for text, att_mask, slots, len_list, perm_idx in test_loader:
        text = text.to(device)
        att_mask = att_mask.to(device)

        # intent_out, slot_out = model(text, att_mask)
        # slot_out = model.crf.decode(slot_out)
        intent_out, slot_out = model.predict(text, att_mask, slots, len_list, perm_idx = perm_idx, device = device)

        true_idx = perm_idx.argsort()
        slot_out = [slot_out[i] for i in true_idx]
        slots = [slots[i] for i in true_idx]
        len_list = [len_list[i] for i in true_idx]
        intent_out = [intent_out[i] for i in true_idx]
        # intent_logits.append(intent_out)
        for intent, sp, sl in zip(intent_out, slot_out, slots):
            assert len(sp) == len(sl)
            intent_pred.append(intent_list[intent])
            # intent_pred.append(intent_list[int(intent.argmax())])
            slot_pred.append([slot_list[out_slot] for out_slot, tgt_slot in zip(sp, sl) if tgt_slot >= 0])

with open('results.csv', 'w') as fw:
    for intent, slot in zip(intent_pred, slot_pred):
        slot_seq = ' '.join(slot)
        fw.write(f'{intent}, {slot_seq}\n')
