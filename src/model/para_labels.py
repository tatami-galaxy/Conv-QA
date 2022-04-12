#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
import string
from typing import List
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F
import re
import json


# In[2]:


def normalize_answer(s: str) -> str:
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


# In[3]:


max_length= 384
pretrained_model = 't5-base'
device = torch.device('cuda')

tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
para_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
rc_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

para_model.load_state_dict(torch.load('/home/ujan/Documents/conv-qa/models/finetuned_weights/para_5.pth'))
rc_model.load_state_dict(torch.load('/home/ujan/Documents/conv-qa/models/finetuned_weights/new_rc_gen6.pth'))

para_model.to(device)
rc_model.to(device)


# In[10]:


class DataClass:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_csv(self, f, output_f):

        answers = []
        rewrites = []
        passages = []
        labels = []

        filepath = self.data_dir+f

        with open(filepath) as fl:
            data = json.load(fl)
      
        for d in data:
            
            para_loss = {}
            losses = []
    
            answers.append(d['answer'])
            rewrites.append(d['rewrite'])
            passages.append(d['passage'])
            
            
            
            passage = tokenizer(d['rewrite'], d['passage'], padding=True, truncation='only_second',
                           max_length=max_length, add_special_tokens=True, return_tensors="pt")
            answer = tokenizer(d['answer'], padding=True, truncation='only_second',
                                  max_length=max_length, add_special_tokens=True, return_tensors="pt")

            psg_input = passage.input_ids.to(device)
            psg_attention = passage.attention_mask.to(device)
            ans_input = answer.input_ids
            ans_input[ans_input == tokenizer.pad_token_id] = -100
            ans_input = ans_input.to(device)

            org_loss = rc_model(input_ids=psg_input, attention_mask=psg_attention, labels=ans_input).loss.item()
            
            
            source = tokenizer(d['rewrite'], truncation=True, max_length=max_length,
                                      add_special_tokens=True, return_tensors="pt")

            input_ids = source.input_ids.to(device)
            attention_masks = source.attention_mask.to(device)
            
            outputs = para_model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=384,
            do_sample=True,
            top_k=120, # 120
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=10)

            for output in outputs:
                line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                passage = tokenizer(line, d['passage'], padding=True, truncation='only_second',
                                   max_length=max_length, add_special_tokens=True, return_tensors="pt")

                psg_input = passage.input_ids.to(device)
                psg_attention = passage.attention_mask.to(device)

                loss = rc_model(input_ids=psg_input, attention_mask=psg_attention, labels=ans_input).loss.item()
                para_loss[line] = loss
                losses.append(loss)

            if any(p < org_loss for p in losses): labels.append(min(para_loss, key=para_loss.get))
            else : labels.append(d['rewrite'])

            

        data = {'answer':answers, 'passage':passages, 'rewrite':rewrites, 'labels':labels}
        df = pd.DataFrame(data)
        df.to_csv(output_f, index=False)


data = DataClass('/home/ujan/Documents/conv-qa/data/interim/')

data.data_csv('qrecc_train.json', 'train.csv')
data.data_csv('qrecc_test.json', 'test.csv')

qrecc_para = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

qrecc_para.save_to_disk("/home/ujan/Desktop/qrecc_para")


# In[ ]:




