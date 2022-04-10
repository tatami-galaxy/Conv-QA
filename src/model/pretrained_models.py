import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F
from os.path import dirname, abspath

#pretrained_model = 't5-base'
#pretrained_model = 'google/t5-v1_1-base'
pretrained_model = 'Vamsi/T5_Paraphrase_Paws'

#tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
#qr_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
#rc_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

root = abspath(__file__)
while root.split('/')[-1] != 'conv-qa':
    root = dirname(root)

#qr_model.save_pretrained(root+'/models/pretrained_models/t5-base')
#tokenizer.save_pretrained(root+'/models/pretrained_models/t5-tokenizer')

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)  
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

model.save_pretrained(root+'/models/pretrained_models/t5-para')
tokenizer.save_pretrained(root+'/models/pretrained_models/para-tokenizer')



