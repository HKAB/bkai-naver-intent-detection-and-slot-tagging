import os
import torch
from torch.utils.data import Dataset

class DataManager:
    
    def __init__(self, root, train, dev, test, max_len = None):
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

        all_text = self.train_data['text'] + self.dev_data['text'] + self.test_data['text']
        tokenized_data, self.word_dict = self.tokenize(all_text)
        self.train_data['text'] = tokenized_data[:self.train_data['size']]
        self.dev_data['text'] = tokenized_data[self.train_data['size']: -self.test_data['size']]
        self.test_data['text'] = tokenized_data[-self.test_data['size']:]

        self.train_data = self.label2id(self.train_data)
        self.dev_data = self.label2id(self.dev_data)


    def get_data(self, name):
        if name == 'train':
            return TextData(self.train_data, self.word_dict, self.slot_label, max_len = self.max_len)
        if name == 'dev':
            return TextData(self.dev_data, self.word_dict, self.slot_label, max_len = self.max_len)
        if name == 'test':
            return TextData(self.test_data, self.word_dict, self.slot_label, is_test=True, max_len = self.max_len)

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

    def tokenize(self, sent_list, word_dict = None):
        if not word_dict:
            word_dict = {'<pad>' : 0}
        tokenized_data = []
        for sent in sent_list:
            tokenized_sent = []
            for w in sent.split():
                if w in word_dict.keys():
                    tokenized_sent.append(word_dict[w])
                else:
                    ind = len(word_dict.keys())
                    word_dict[w] = ind
                    tokenized_sent.append(ind)

            tokenized_data.append(tokenized_sent)
        
        return tokenized_data, word_dict

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
    
    def __init__(self, data, word_dict, slot_labels, is_test = False, max_len = None) -> None:
        self.data = data
        self.is_test = is_test
        self.word_dict = word_dict
        self.slot_labels = slot_labels
        self.max_len = max_len

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, index):
        if self.is_test:
            return self.data['text'][index]
        else:
            return self.data['text'][index], self.data['slots'][index], self.data['intents'][index]

    def collate_fn(self, batch):
        if self.is_test:
            text = batch
            text, len_list = self.add_pad(text, self.word_dict['<pad>'])
            text = torch.tensor(text)
            len_list = torch.tensor(len_list)
            len_list, perm_idx = len_list.sort(descending = True)
            text = text[perm_idx]
            return text, len_list, perm_idx
        else:
            text, slots, intents = list(zip(*batch))
            text, len_list = self.add_pad(text, self.word_dict['<pad>'])
            slots, _ = self.add_pad(slots, self.slot_labels.index('O'))
            text, slots, intents = torch.tensor(text), torch.tensor(slots), torch.tensor(intents)
            len_list = torch.tensor(len_list)
            len_list, perm_idx = len_list.sort(descending = True)
            text = text[perm_idx]
            slots = slots[perm_idx]
            intents = intents[perm_idx]
            return text, slots, intents, len_list, perm_idx

    def add_pad(self, data, pad_token):
        len_list = [len(s) for s in data]
        max_len = max(len_list) if self.max_len is None else self.max_len
        padded_data = []
        for datum, sent_len in zip(data, len_list):
            datum.extend([pad_token] * (max_len - sent_len))
            padded_data.append(datum)
        
        return padded_data, len_list
            

    