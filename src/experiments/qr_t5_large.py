import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F

model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-large')
model.save_pretrained('t5_v1_1-large')
