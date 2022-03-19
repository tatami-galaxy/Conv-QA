import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F

class DataClass:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def data_csv(self, f, output):

    contexts = []
    questions = []
    rewrites = []

    filepath = self.data_dir+f

    with open(filepath) as fl:
      data = json.load(fl)
      
      for d in data:
        contexts.append(d['context'])
        questions.append(d['question'])
        rewrites.append(d['rewrite'])

      data = {'context':contexts, 'question':questions, 'rewrite':rewrites}
      df = pd.DataFrame(data)
      df.to_csv(output, index=False)


data = DataClass('/home/ujan/Documents/conv-qa/data/interim/')

data.data_csv('qrecc_train.json', 'train.csv')
data.data_csv('qrecc_test.json', 'test.csv')

qrecc = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

max_length= 384
batch_size = 8
dim = 768 # change BERT hidden size to change

pretrained_model = 't5-large'

tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

print('Model loaded')

def tokenize_dataset(batch):
  contexts = tokenizer(batch['context'], batch['question'], padding='max_length', truncation='only_first', max_length=max_length, add_special_tokens=True)
  rewrites = tokenizer(batch['rewrite'], padding='max_length', truncation=True, max_length=max_length, add_special_tokens=True)

  batch['ctx_input_ids'] = contexts.input_ids
  batch['rwrt_input_ids'] = rewrites.input_ids


  batch['ctx_attention_mask'] = contexts.attention_mask
  batch['rwrt_attention_mask'] = rewrites.attention_mask

  return batch


# removing examples with no context
qrecc = qrecc.filter(lambda x: isinstance(x['context'], str) and isinstance(x['rewrite'], str))


# tokenizing
dataset = qrecc.map(
    tokenize_dataset,
    batch_size = batch_size,
    batched=True,
    remove_columns=['context', 'question', 'rewrite']
)


dataset.set_format(
    type='torch', columns=['ctx_input_ids', 'rwrt_input_ids', 'ctx_attention_mask', 'rwrt_attention_mask'],)

print('Tokenization done')

train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size)

def valid_loss():

  val_loss = 0
  idx = 0

  for batch in test_loader:

    ctx_input = batch['ctx_input_ids'].to(device) # QR input
    ctx_attention = batch['ctx_attention_mask'].to(device)

    rwrt_input = batch['rwrt_input_ids'].to(device)
    rwrt_input[rwrt_input == tokenizer.pad_token_id] = -100
    rwrt_input = rwrt_input.to(device)

    loss = model(input_ids=ctx_input, attention_mask=ctx_attention, labels=rwrt_input).loss
    val_loss += loss.item()

    del ctx_input, ctx_attention, rwrt_input, loss

    idx += 1

  return val_loss/idx


num_epochs = 5

device = torch.device('cuda')
model.to(device)

# model.load_state_dict(torch.load('/storage/qrecc/models/qr/qr_gen3.pth'))

model.train()

optim = optimizer = Adafactor(
    model.parameters(),
    lr=1e-5,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)

print('Training start')

for epoch in range(num_epochs):

  epoch_loss = 0

  for batch in train_loader:

    ctx_input = batch['ctx_input_ids'].to(device) # QR input
    ctx_attention = batch['ctx_attention_mask'].to(device)

    rwrt_input = batch['rwrt_input_ids'].to(device)
    rwrt_input[rwrt_input == tokenizer.pad_token_id] = -100 # tokens with indices set to -100 are ignored (masked)
    rwrt_input = rwrt_input.to(device)

    loss = model(input_ids=ctx_input, attention_mask=ctx_attention, labels=rwrt_input).loss
    epoch_loss += loss.item()

    loss.backward()
    optim.step()
    optim.zero_grad()


    del ctx_input, ctx_attention, rwrt_input, loss

  print('Train loss after epoch {} : {}'.format(epoch+1, epoch_loss/len(train_loader)))
  model.eval()
  print('Valid loss after epoch {} : {}'.format(epoch+1, valid_loss()))
  print('\n')
  model.train()
  torch.save(model.state_dict(), '/home/ujan/Documents/qr_gen_large'+str(epoch+1)+'.pth')
