from transformers.generation_utils import top_k_top_p_filtering

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from vncorenlp import VnCoreNLP
import numpy as np

rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
mask_model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")

def get_same_word_id(sent):
    word_len = [len(i.split('_')) for i in sent.split()]
    # word_len_sum = np.cumsum(word_len)
    # same_word_id = [i + j  for i, l in enumerate(word_len) for j in range(l) if l > 1]
    stack = 0
    same_word_id = []
    for i, l in enumerate(word_len):
        if l != 1:
            for j in range(1, l):
                same_word_id.append(i + j + stack)
            stack += l -1
    return same_word_id

def get_span(label, same_word_id):
    span_list = []
    inner = False
    sub_id = 0
    for i, l in enumerate(label):
        if i in same_word_id:
            sub_id -= 1
            continue
        elif l == 'O':
            if inner:
                span_list.append(curr_span)
                inner = False 
        elif l[0] == 'B':
            if inner:
                span_list.append(curr_span)
            curr_span = {'label': l[2:], 'start': i + sub_id, 'end': i + sub_id}
            inner = True
        elif l[0] == 'I':
            curr_span['end'] += 1
    if label[-1] != 'O':
        span_list.append(curr_span)
    return span_list

def add_mask(token_ids, span, mask_id):
    token_ids[:, span['start'] + 1:span['end'] + 2] = mask_id
    return token_ids

def nucleus_sampling(model, inputs, span, mask_id, p = 0.95):
    for i in range(span['start'] + 1, span['end'] + 2):
        logits = model(**inputs).logits
        mask_logits = logits[:, i, :]
        filtered_logits = top_k_top_p_filtering(mask_logits.exp(), top_p = p)[0]
        filtered_logits[filtered_logits.isinf()] = 0
        # if filtered_logits.isnan().sum():
        #     print(filtered_logits[filtered_logits.isnan()])
        # if filtered_logits.isinf().sum():
        #     print(filtered_logits[filtered_logits.isinf()])
        # if (filtered_logits < 0).any():
        #     print(filtered_logits[filtered_logits < 0], filtered_logits[filtered_logits < 0].shape)
        # print(filtered_logits)
        selected_id = filtered_logits.multinomial(1)
        inputs.input_ids[0, i] = selected_id 
    return inputs


if __name__ == '__main__':

    with open('training_data/seq.in', 'r') as f:
        text = f.read().splitlines()
    text = [' '.join(rdrsegmenter.tokenize(sent)[0]) for sent in text]

    with open('training_data/seq.out', 'r') as f:
        label = f.read().splitlines()

    
    content = []
    for j in range(len(text)):
        try:
            same_word_id = get_same_word_id(text[j])
            span_list = get_span(label[j].split(), same_word_id)
            token = tokenizer(text[j], return_tensors = 'pt')
            token.input_ids = add_mask(token.input_ids, span_list[0], tokenizer.mask_token_id)
            out = nucleus_sampling(mask_model, token, span_list[0], tokenizer.mask_token_id)
            aug = tokenizer.decode(out.input_ids[0])
            aug = ' '.join([w if span_list[0]['start'] + 1 > i or span_list[0]['end'] + 1 < i else w.upper() for i, w in enumerate(aug.split())])

            content.append(aug)
        except:
            pass

    with open('augment.txt', 'w') as f:
        f.write('\n'.join(content))