#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[32]:


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


# In[33]:


filename = "/home/ujan/Documents/conv-qa/data/interim/quora_duplicate_questions.tsv"

question_pairs = pd.read_csv(filename, sep='\t')
question_pairs.drop(['qid1', 'qid2'], axis = 1,inplace = True)

question_pairs_correct_paraphrased = question_pairs[question_pairs['is_duplicate']==1]

question_pairs_correct_paraphrased.drop(['id', 'is_duplicate'], axis = 1,inplace = True)
train, test = train_test_split(question_pairs_correct_paraphrased, test_size=0.1)

train.to_csv('/home/ujan/Documents/conv-qa/data/interim/Quora_Paraphrasing_train.csv', index = False)
test.to_csv('/home/ujan/Documents/conv-qa/data/interim/Quora_Paraphrasing_val.csv', index = False)


# In[34]:


quora= load_dataset('csv', data_files={'train': '/home/ujan/Documents/conv-qa/data/interim/Quora_Paraphrasing_train.csv',
                                       'test': '/home/ujan/Documents/conv-qa/data/interim/Quora_Paraphrasing_val.csv'})


# In[35]:


max_length= 384
batch_size = 16

pretrained_model = 't5-base'

tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model)


# In[36]:


def tokenize_dataset(batch):
    source = tokenizer(batch['question1'], padding='max_length', truncation=True, max_length=max_length, add_special_tokens=True)
    para = tokenizer(batch['question2'], padding='max_length', truncation=True, max_length=max_length, add_special_tokens=True)

    batch['src_input_ids'] = source.input_ids
    batch['para_input_ids'] = para.input_ids
 

    batch['src_attention_mask'] = source.attention_mask
    batch['para_attention_mask'] = para.attention_mask

    return batch

def sanitize(x):
    x['question1'] = normalize_answer(x['question1'])
    x['question2'] = normalize_answer(x['question2'])
    
    return x

    
# removing empty examples
quora = quora.filter(lambda x: isinstance(x['question1'], str) and isinstance(x['question2'], str))

# sanitize
quora = quora.map(sanitize)

# tokenizing
dataset = quora.map(tokenize_dataset, batch_size = batch_size, batched=True, remove_columns=['question1', 'question2'])


dataset.set_format(
    type='torch', columns=['src_input_ids', 'para_input_ids', 'src_attention_mask', 'para_attention_mask'],)


# In[22]:


train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size)


# In[23]:


def valid_loss():
  
    val_loss = 0
    idx = 0

    for batch in test_loader:

        src_input = batch['src_input_ids'].to(device) 
        src_attention = batch['src_attention_mask'].to(device)

        para_input = batch['para_input_ids'].to(device) 
        para_input[para_input == tokenizer.pad_token_id] = -100
        para_input = para_input.to(device)

        loss = model(input_ids=src_input, attention_mask=src_attention, labels=para_input).loss
        val_loss += loss.item()

        idx += 1

    return val_loss/idx


# In[24]:


num_epochs = 5

device = torch.device('cuda')
model.to(device)

# model.load_state_dict(torch.load('/storage/qrecc/models/qr/qr_gen3.pth'))

model.train()

optim = optimizer = Adafactor(
    model.parameters(),
    lr=3e-4, # 1e-5
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)

for epoch in range(num_epochs):
  
    epoch_loss = 0

    for batch in train_loader:

        src_input = batch['src_input_ids'].to(device) 
        src_attention = batch['src_attention_mask'].to(device)

        para_input = batch['para_input_ids'].to(device) 
        para_input[para_input == tokenizer.pad_token_id] = -100
        para_input = para_input.to(device)

        loss = model(input_ids=src_input, attention_mask=src_attention, labels=para_input).loss
        epoch_loss += loss.item() 

        loss.backward()
        optim.step()
        optim.zero_grad()
        

    print('Train loss after epoch {} : {}'.format(epoch+1, epoch_loss/len(train_loader)))
    model.eval()
    print('Valid loss after epoch {} : {}'.format(epoch+1, valid_loss()))
    print('\n')
    model.train()
    torch.save(model.state_dict(), '/home/ujan/Documents/'+str(epoch+1)+'.pth')


# In[ ]:




