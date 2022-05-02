import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DataManager:
    
    def __init__(self, root, train, dev, test, max_len = None, pretrained = None):
        self.root = root
        self.train_path = os.path.join(root, train)
        self.dev_path = os.path.join(root, dev)
        self.test_path = os.path.join(root, test)
        self.max_len = max_len
        self.train_data = self.__read_data(self.train_path)
        self.dev_data = self.__read_data(self.dev_path)
        self.test_data = self.__read_data(self.test_path, is_test = True)
        self.intent_label = self.train_data['intent_label']
        self.slot_label = self.train_data['slot_label'][2:]
        self.pad_label_id = -1
        self.pad_id = 0

        self.pretrained = pretrained
        if pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
            self.pad_id = self.tokenizer.pad_token_id

            self.train_data['text'], self.train_data['slots'] = self.trf_tokenize(self.train_data['text'], self.train_data['slots'])
            self.dev_data['text'], self.dev_data['slots'] = self.trf_tokenize(self.dev_data['text'], self.dev_data['slots'])
            self.test_data['text'], self.test_data['slots'] = self.trf_tokenize(self.test_data['text'])
            self.train_data['intents'] = self.intent2id(self.train_data['intents'])
            self.dev_data['intents'] = self.intent2id(self.dev_data['intents'])
            self.word_dict = {}

        else:
            all_text = self.train_data['text'] + self.dev_data['text'] + self.test_data['text']
            tokenized_data, self.word_dict = self.tokenize(all_text)
            self.train_data['text'] = tokenized_data[:self.train_data['size']]
            self.dev_data['text'] = tokenized_data[self.train_data['size']: -self.test_data['size']]
            self.test_data['text'] = tokenized_data[-self.test_data['size']:]

            self.train_data = self.label2id(self.train_data)
            self.dev_data = self.label2id(self.dev_data)
            self.test_data['slots'] = [[self.pad_label_id + 1] * len(s) for s in self.test_data['text']]


    def get_data(self, name):
        if name == 'train':
            return TextData(self.train_data, self.pad_id, self.pad_label_id, max_len = self.max_len)
        if name == 'dev':
            return TextData(self.dev_data, self.pad_id, self.pad_label_id, max_len = self.max_len)
        if name == 'test':
            return TextData(self.test_data, self.pad_id, self.pad_label_id, is_test=True, max_len = self.max_len)

    def __read_data(self, folder, is_test = False):
        data = {}
        with open(os.path.join(folder, 'seq.in'), 'r') as fr:
            data['text'] = fr.read().splitlines()
        with open(os.path.join(folder, 'slot_label.txt'), 'r') as fr:
            data['slot_label'] = fr.read().splitlines()
        with open(os.path.join(folder, 'intent_label.txt'), 'r') as fr:
            data['intent_label'] = fr.read().splitlines()
        data['size'] = len(data['text'])

        if not is_test:
            with open(os.path.join(folder, 'seq.out'), 'r') as fr:
                data['slots'] = fr.read().splitlines()

            with open(os.path.join(folder, 'label'), 'r') as fr:
                data['intents'] = fr.read().splitlines()

        return data

    def trf_tokenize(self, sent_list, slot_list = None):
        if slot_list is not None:
            assert len(sent_list) == len(slot_list)
        input_ids = []
        slot_ids = [] 
        for i in range(len(sent_list)):
            sent  = sent_list[i].split()
            sent_ids = [self.tokenizer.cls_token_id]
            sent_slot_ids = [self.pad_label_id]

            if slot_list is not None:
                sent_slots = slot_list[i].split()
                assert len(sent) == len(sent_slots)

            for j in range(len(sent)):
                word = sent[j]
                tokens = self.tokenizer.tokenize(word)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                sent_ids.extend(token_ids)

                if slot_list is not None:
                    slot = sent_slots[j]
                    sent_slot_ids.extend([self.slot_label.index(slot)] + [self.pad_label_id] * (len(tokens) - 1))
                else:
                    sent_slot_ids.extend([self.pad_label_id + 1] + [self.pad_label_id] * (len(tokens) - 1))

            sent_ids.append(self.tokenizer.sep_token_id)
            input_ids.append(sent_ids)

            sent_slot_ids.append(self.pad_label_id)
            assert len(sent_ids) == len(sent_slot_ids)
            slot_ids.append(sent_slot_ids)

        return input_ids, slot_ids


    def tokenize(self, sent_list, word_dict = None):
        if not word_dict:
            word_dict = {'<pad>' : 0}
        input_ids = []
        for sent in sent_list:
            tokenized_sent = []
            for w in sent.split():
                if w in word_dict.keys():
                    tokenized_sent.append(word_dict[w])
                else:
                    ind = len(word_dict.keys())
                    word_dict[w] = ind
                    tokenized_sent.append(ind)

            input_ids.append(tokenized_sent)
        
        return input_ids, word_dict

    def slot2id(self, labels):
        label_ids = []
        for item in labels:
            sent_labels = [self.slot_label.index(label) for label in item.split()]
            label_ids.append(sent_labels)

        return label_ids

    def intent2id(self, labels):
        return [self.intent_label.index(l) for l in labels]

    def label2id(self, data):
        data['slots'] = self.slot2id(data['slots'])
        data['intents'] = self.intent2id(data['intents'])
        return data

class TextData(Dataset):
    
    def __init__(self, data, pad_id, pad_label_id, is_test = False, max_len = None) -> None:
        self.data = data
        self.is_test = is_test
        self.pad_id = pad_id
        self.pad_label_id = pad_label_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, index):
        if self.is_test:
            return self.data['text'][index], self.data['slots'][index]
        else:
            return self.data['text'][index], self.data['slots'][index], self.data['intents'][index]

    def collate_fn(self, batch):
        if self.is_test:
            text, slot_mask = list(zip(*batch))
            text, len_list, att_mask = self.add_pad(text, self.pad_id, return_mask = True)
            slot_mask, _ = self.add_pad(slot_mask, self.pad_label_id)

            text, att_mask, slot_mask = torch.tensor(text), torch.tensor(att_mask), torch.tensor(slot_mask)
            len_list = torch.tensor(len_list)
            len_list, perm_idx = len_list.sort(descending = True)
            text = text[perm_idx]
            slot_mask = slot_mask[perm_idx]
            att_mask = att_mask[perm_idx]
            return text, att_mask, slot_mask, len_list, perm_idx
        else:
            text, slots, intents = list(zip(*batch))
            text, len_list, att_mask = self.add_pad(text, self.pad_id, return_mask = True)
            slots, _ = self.add_pad(slots, self.pad_label_id)
            text, att_mask, slots, intents = torch.tensor(text), torch.tensor(att_mask), torch.tensor(slots), torch.tensor(intents)
            len_list = torch.tensor(len_list)
            len_list, perm_idx = len_list.sort(descending = True) # sort by len for pack_padded_sequence
            text = text[perm_idx]
            att_mask = att_mask[perm_idx]
            slots = slots[perm_idx]
            intents = intents[perm_idx]
            return text, att_mask, slots, intents, len_list, perm_idx

    def add_pad(self, data, pad_token, return_mask = False):
        len_list = [len(s) for s in data]
        max_len = max(len_list) if self.max_len is None else self.max_len
        padded_data = []
        att_mask = []
        for datum, sent_len in zip(data, len_list):
            pad_len = max_len - sent_len
            datum.extend([pad_token] * pad_len)
            padded_data.append(datum)
            if return_mask:
                att_mask.append([1] * sent_len + [0] * pad_len)
        
        if return_mask:
            return padded_data, len_list, att_mask
        else:
            return padded_data, len_list
            

    