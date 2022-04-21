import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from scipy import stats

from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        is_saved = False
        best_sementic_frame_acc = -100
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    #     self.evaluate("dev")

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            # save the best epoch
            epoch_result = self.evaluate("dev")
            if (epoch_result['sementic_frame_acc'] < best_sementic_frame_acc):
                self.save_model()
                is_saved = True
                best_sementic_frame_acc = epoch_result['sementic_frame_acc']

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        if (not is_saved):
            self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = []
        slot_preds = []
        out_intent_label_ids = []
        out_slot_labels_ids = []

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs, teacher_forcing=False)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            seq_lens = torch.sum(inputs["attention_mask"], dim=1)
            # Intent prediction
            # if intent_preds is None:
            #     # intent_preds: (batch_size*total_word_len x num_intent_labels)
            #     intent_preds = np.argmax(intent_logits.detach().cpu().numpy(), dim=1)
            #     out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            # else:
            # print(intent_logits)
            # print(np.argmax(intent_logits.detach().cpu().numpy(), axis=1))

            # print(intent_logits)

            start_pos = 0
            for i in range(len(seq_lens)):
                sent_intent = intent_logits.detach().cpu().numpy()[start_pos:start_pos + seq_lens[i]]
                intent_preds.append(stats.mode(np.argmax(sent_intent, axis=1)).mode[0])
                start_pos = start_pos + seq_lens[i]

            out_intent_label_ids.extend(inputs['intent_label_ids'].detach().cpu().numpy())

            # Slot prediction
            # if slot_preds is None:
            # print(slot_logits)
            if self.args.use_crf:
                # fix this later
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = slot_preds.append(self.model.crf.decode(slot_logits))
            else:
                # slot_logits: (batch_size*total_word_len x num_slot_labels)
                start_pos = 0
                for i in range(len(seq_lens)):
                    sent_slot = slot_logits.detach().cpu().numpy()[start_pos:start_pos + seq_lens[i]]
                    slot_preds.append(np.argmax(sent_slot, axis=1))
                    start_pos = start_pos + seq_lens[i]
                # slot_preds = slot_logits.detach().cpu().numpy()

            padded_out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            for i in range(len(padded_out_slot_labels_ids)):
                out_slot_labels_ids.append(padded_out_slot_labels_ids[i][:seq_lens[i]])
        # else:
            #     if self.args.use_crf:
            #         slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
            #     else:
            #         slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

            #     out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        # intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        # if not self.args.use_crf:
        #     slot_preds = np.argmax(slot_preds, axis=1)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(len(out_slot_labels_ids))]
        slot_preds_list = [[] for _ in range(len(out_slot_labels_ids))]

        for i in range(len(out_slot_labels_ids)):
            for j in range(len(out_slot_labels_ids[i])):
                if out_slot_labels_ids[i][j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # print(slot_logits)
        # print(slot_preds)
        # print(slot_preds_list)

        # print(slot_preds_list)
        # print(out_slot_label_list)
        # print(out_intent_label_ids)
        # print(len(intent_preds), len(out_intent_label_ids), len(slot_preds_list), len(out_slot_label_list))
        intent_preds = np.array(intent_preds)
        out_intent_label_ids = np.array(out_intent_label_ids)
        # print(slot_preds_list, out_slot_label_list)
        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
