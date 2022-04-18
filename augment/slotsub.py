import random
import pickle
import os
import re
from preprocess import preprocess

def get_span_and_ent(text, label):
    assert len(label) == len(text)
    span_list = []
    inner = False
    for i, (l, w) in enumerate(zip(label, text)):
        if l == 'O':
            if inner:
                span_list.append(curr_span)
                inner = False 
        elif l[0] == 'B':
            if inner:
                span_list.append(curr_span)
            curr_span = {'label': l[2:], 'start': i, 'end': i, 'entity': w}
            inner = True
        elif l[0] == 'I':
            curr_span['end'] += 1
            curr_span['entity'] += f' {w}'
    if label[-1] != 'O':
        span_list.append(curr_span)
    return span_list

def get_ent_dict():
    dict_file = 'augment/ent_dict.pickle'
    if os.path.exists(dict_file):
        with open(dict_file, 'rb') as f:
            ent_dict = pickle.load(f)

            return ent_dict
    else:
        with open('data/training_data/seq.in', 'r') as fr:
            text = fr.read().splitlines()
        with open('data/training_data/seq.out', 'r') as fr:
            label = fr.read().splitlines()

        text = [preprocess(s) for s in text]
        ent_list = [get_span_and_ent(text[i].split(), label[i].split()) for i in range(len(text))]
        ent_list = sum(ent_list, [])
        with open('data/training_data/slot_label.txt', 'r') as f:
            label_set = f.read().splitlines()[3:]
            label_set = set([i[2:] for i in label_set])
        ent_dict = {l:[] for l in label_set}
        for ent in ent_list:
            ent_dict[ent['label']].append(ent['entity'])
        for k in ent_dict.keys():
            ent_dict[k] = list(set(ent_dict[k]))
            # if 'sysnumber' in k:
            #     ent_dict[k] = [str(i) for i in range(10)]
            # else:
            #     ent_dict[k] = list(set(ent_dict[k]))
        with open('augment/ent_dict.pickle', 'wb') as p:
            pickle.dump(ent_dict, p)
        return ent_dict

def expand_tag(label, length):
    return [f'B-{label}'] + [f'I-{label}'] * (length - 1)

def change_numbers(text):
    # numbers = re.findall(r'\d+', a)
    # new_nums = [random.randint(1, 9) for _ in range(len(numbers))]
    # for old_num, new_num in zip(numbers, new_nums):
    #     text = re.sub(fr'\b{old_num}\b', fr'{new_num}', a)
    percent = random.randint(1, 99)
    text = re.sub(r'[0-9]* phần trăm', fr'{percent} phần trăm', text)
    return text
    
    
    

def slotsub(sent, label, num_samples = 1):
    text_list = sent.split()
    label_list = label.split()
    span = get_span_and_ent(text_list, label_list)
    if len(span) == 0:
        return None

    ent_dict = get_ent_dict()
    # selected_span = {'label': 'sysnumber'}
    # while not 'sysnumber' in selected_span['label']:
    selected_span_id = random.choice(range(len(span)))
    selected_span = span[selected_span_id]
    substitute = [selected_span['label'] for _ in range(num_samples)]
    while sum([s == selected_span['label'] for s in substitute]) == num_samples:
        substitute = random.sample(ent_dict[selected_span['label']], k = num_samples)
    new_sent = [text_list[:selected_span['start']] + s.split() + text_list[selected_span['end'] + 1:] for s in substitute]
    raw_label = [expand_tag(selected_span['label'], len(s.split())) for s in substitute]
    new_label = [label_list[:selected_span['start']] + l + label_list[selected_span['end'] + 1:] for l in raw_label]
    # new_span = []
    # for s in substitute:
    #     selected_span['entity'] = s
    #     selected_span['end'] = selected_span['start'] + len(s.split())
    #     new_span.append(selected_span)
    new_sent = [' '.join(s) for s in new_sent]
    new_sent = [change_numbers(s) for s in new_sent]
    new_label = [' '.join(l) for l in new_label]
    return new_sent, new_label

if __name__ == '__main__':
    text = 'bạn có thể tăng giúp mình bóng chùm thứ 3 lên mức 21 phần trăm được không'
    label = 'O O O O O O B-devicedevice I-devicedevice I-devicedevice I-devicedevice O O B-change-valuesyspercentage I-change-valuesyspercentage I-change-valuesyspercentage O O'
    print(slotsub(text, label))