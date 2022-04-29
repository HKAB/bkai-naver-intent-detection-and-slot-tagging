import json
from dataloader import *
from models import *
from torch.utils.data import DataLoader

save_dir = 'save/v1'
with open(os.path.join(save_dir, 'config.json'), 'r') as f:
    args = json.load(f)
class HP:
    def __init__(self, **entries):
        self.__dict__.update(entries)
args = HP(**args)

device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
dataset = DataManager(args.data_dir, args.train_folder, args.dev_folder, args.test_folder, max_len=args.max_len)
num_word = len(dataset.word_dict.keys())
num_slot = len(dataset.slot_label)
num_intent = len(dataset.intent_label)
test_data = dataset.get_data('test')
test_loader = DataLoader(test_data, batch_size = args.dev_batch_size, collate_fn = test_data.collate_fn, shuffle = False, pin_memory = True)

embedding = WordEmbedding(num_word, args.emb_dim)
slot_model = SlotModel(embedding, args.emb_dim, args.hidden_dim, num_slot, args.dropout, args.max_len).to(device)
embedding = WordEmbedding(num_word, args.emb_dim)
intent_model = IntentModel(embedding, args.emb_dim, args.hidden_dim, num_intent, args.dropout, args.max_len).to(device)
slot_model.load_state_dict(torch.load(os.path.join(save_dir, 'slot.pth')))
intent_model.load_state_dict(torch.load(os.path.join(save_dir, 'intent.pth')))

slot_list = dataset.slot_label
intent_list = dataset.intent_label

slot_model.eval()
intent_model.eval()
slot_pred = []
intent_pred = []
with torch.no_grad():
    for text, len_list, perm_idx in test_loader:
        text = text.to(device)

        intent_feat = intent_model.encode(text, len_list)
        intent_share = intent_feat.clone().detach()
        slot_feat = slot_model.encode(text, len_list)
        slot_share = slot_feat.clone().detach()
        
        intent_out = intent_model.decode(intent_feat, slot_share, len_list)
        slot_out = slot_model.decode(slot_feat, intent_share, len_list)
        slot_out = slot_model.crf.decode(slot_out)

        true_idx = perm_idx.argsort()
        slot_out = [slot_out[i] for i in true_idx]
        len_list = [len_list[i] for i in true_idx]
        intent_out = [intent_out[i] for i in true_idx]
        # intent_logits.append(intent_out)
        for intent, sp, length in zip(intent_out, slot_out, len_list):
            intent_pred.append(intent_list[int(intent.argmax())])
            slot_pred.append([slot_list[sp[i]] for i in range(length)])

with open('results.csv', 'w') as fw:
    for intent, slot in zip(intent_pred, slot_pred):
        slot_seq = ' '.join(slot)
        fw.write(f'{intent}, {slot_seq}\n')