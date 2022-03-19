import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AlbertTokenizer, AlbertModel
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F
import collections
from typing import List
from sklearn.metrics import f1_score

class DataClass:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def data_csv(self, f, output):

    answers = []
    rewrites = []
    passages = []
    contexts = []

    filepath = self.data_dir+f

    with open(filepath) as fl:
      data = json.load(fl)

      for d in data:
        answers.append(d['answer'])
        rewrites.append(d['rewrite'])
        passages.append(d['passage'])
        contexts.append(d['context']+' '+d['question'])

      data = {'answer':answers, 'passage':passages, 'rewrite':rewrites, 'context':contexts}
      df = pd.DataFrame(data)
      df.to_csv(output, index=False)


data = DataClass('/home/ujan/Documents/conv-qa/data/interim/')

data.data_csv('qrecc_train.json', 'train.csv')
data.data_csv('qrecc_test.json', 'test.csv')

qrecc = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

max_length= 384
batch_size = 4
pretrained_model = 't5-base'

t5_tokenizer = T5Tokenizer.from_pretrained(pretrained_model)

t5_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

print(qrecc['train'][59])
quit()

rewrite = 'did ralph waldo emerson give many lectures'
#passage  = t5_tokenizer(qrecc['train'][41]['rewrite'], qrecc['train'][41]['passage'], add_special_tokens=True, return_tensors="pt").input_ids
passage  = t5_tokenizer(rewrite, qrecc['train'][41]['passage'], add_special_tokens=True, return_tensors="pt").input_ids
answer = t5_tokenizer(qrecc['train'][41]['answer'], add_special_tokens=True, return_tensors="pt").input_ids

device = torch.device('cuda')
t5_model.to(device)
t5_model.load_state_dict(torch.load('/home/ujan/Documents/conv-qa/models/finetuned_weights/rc_gen5.pth'))

passage = passage.to(device)
answer = answer.to(device)

loss = t5_model(input_ids=passage, labels=answer).loss
print(loss)
