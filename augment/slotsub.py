import random
import pickle
import os
from webbrowser import get

def get_ent_dict():
    dict_file = 'augment/ent_dict.pickle'
    if os.path.exists(dict_file):
        with open(dict_file, 'rb') as f:
            ent_dict = pickle.load(f)

            return ent_dict

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

def slotsub(sent, label, num_samples = 1):
    text_list = sent.split()
    label_list = label.split()
    span = get_span_and_ent(text_list, label_list)

    ent_dict = get_ent_dict()

    selected_span_id = random.choice(range(len(span)))
    selected_span = span[selected_span_id]
    substitute = random.sample(ent_dict[selected_span['label']], k = num_samples)
    new_sent = [text_list[:selected_span['start']] + s.split() + text_list[selected_span['end'] + 1:] for s in substitute]
    new_span = []
    for s in substitute:
        selected_span['entity'] = s
        selected_span['end'] = selected_span['start'] + len(s.split())
        new_span.append(selected_span)

    return new_sent, new_span

if __name__ == '__main__':
    text = 'bạn có thể tăng giúp mình bóng chùm thứ 3 lên mức 21 phần trăm được không'
    label = 'O O O O O O B-devicedevice I-devicedevice I-devicedevice I-devicedevice O O B-change-valuesyspercentage I-change-valuesyspercentage I-change-valuesyspercentage O O'
    print(slotsub(text, label))