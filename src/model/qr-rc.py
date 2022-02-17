import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import Adafactor
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from os.path import dirname, abspath
from dataclasses import dataclass, field
from collections import namedtuple
from utils import *

@dataclass
class Options:  # class for storing hyperparameters and other options

    max_length : int = 384  # use interim data if changed 
    batch_size : int = 4
    embed_dim : int = 768  # typical base model embedding dimension
    pretrained_model_name : str = 't5-base'
    act_vocab_size : int = 32100  # get from tokenizer
    num_epochs : int = 3

    # adafactor hyperparameters
    lr : float = 1e-5
    eps: tuple = (1e-30, 1e-3)
    clip_threshold : float = 1.0
    decay_rate : float = -0.8
    beta1 : float = None
    weight_decay ; float = 0.0
    relative_step : bool = False
    scale_parameter : bool = False
    warmup_init : bool = False

    # gumbel softmax
    tau : float = 1.0

    # directories
    root : str = field(init=False)
    pretrained_model : str = field(init=False)
    qr_finetuned : str = field(init=False)
    rc_finetuned : str = field(init=False)

    # add methods to init dataclass attributes here
    def __post__init(self):
        self.root = self.get_root_dir()
        self.pretrained_model = self.root + '/models/pretrained_models/t5-base'
        self.qr_finetuned = self.root + '/models/finetuned_weights/qr_gen4.pth'
        self.rc_finetuned = self.root + '/models/finetuned_weights/rc_gen5.pth'
        

    def get_root_dir(self):
        root = abspath(__file__)
        while root.split('/')[-1] != 'conv-qa':
            root = dirname(root)
        return root

         

class End2End(nn.Module):

    def __init__(self, options, device):  
        super().__init__()        

        # load T5 models
        self.qr_model = T5ForConditionalGeneration.from_pretrained(options.pretrained_model)
        self.rc_model = T5ForConditionalGeneration.from_pretrained(options.pretrained_model)

        self.qr_model.to(device)
        self.rc_model.to(device)
        
        # load finetuned weights
        self.qr_model.load_state_dict(torch.load(options.qr_finetuned, map_location=device))  # comment out to only use pretraining
        self.rc_model.load_state_dict(torch.load(options.rc_finetuned, map_location=device))  # comment out to only use pretraining

        self.qr_model.train()
        self.rc_model.train()


    def forward(self, batch):




# device = torch.device('cpu')
device = torch.device('cuda')

max_length = 384
batch_size = 4
embed_dim = 768

pretrained_model = 't5-base'

root = abspath(__file__)
while root.split('/')[-1] != 'conv-qa':
    root = dirname(root)


tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
qr_model = T5ForConditionalGeneration.from_pretrained(root+'/models/pretrained_models/t5-base')
rc_model = T5ForConditionalGeneration.from_pretrained(root+'/models/pretrained_models/t5-base')

qr_model.to(device)
rc_model.to(device)

qr_model.train()
rc_model.train()

#qr_model.load_state_dict(torch.load(root+'/models/finetuned_weights/qr_gen4.pth', map_location=torch.device('cpu')))
#rc_model.load_state_dict(torch.load(root+'/models/finetuned_weights/rc_gen5.pth', map_location=torch.device('cpu')))


#qr_model.load_state_dict(torch.load(root+'/models/finetuned_weights/qr_gen4.pth'))
#rc_model.load_state_dict(torch.load(root+'/models/finetuned_weights/rc_gen5.pth'))

act_vocab_size = len(tokenizer.get_vocab())
num_epochs = 2

optim = Adafactor(
    list(qr_model.parameters())+list(rc_model.parameters()),
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

dataset = load_from_disk(root+'/data/processed/dataset/')
dataset.set_format(type='torch', columns=['ctx_input_ids', 'rwrt_input_ids', 'psg_input_ids',
                   'ans_input_ids', 'ctx_attention_mask', 'rwrt_attention_mask', 'psg_attention_mask'],)

train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size)

def roll_by_gather(mat, dim, shifts:torch.LongTensor):
    # assumes 2D array
    n_rows, n_cols = mat.shape
    
    if dim == 0:
        #print(mat)
        arange1 = torch.arange(n_rows).view((n_rows, 1)).repeat((1, n_cols)).to(device)
        #print(arange1)
        arange2 = (arange1 - shifts) % n_rows
        #print(arange2)
        return torch.gather(mat, 0, arange2)
    elif dim == 1:
        arange1 = torch.arange(n_cols).view((1,n_cols)).repeat((n_rows,1)).to(device)
        #print(arange1)
        arange2 = (arange1 - shifts) % n_cols
        #print(arange2)
        return torch.gather(mat, 1, arange2)


def forward(batch):
    # context + question input
    ctx_input = batch['ctx_input_ids'].to(device)  # QR input
    ctx_attention = batch['ctx_attention_mask'].to(device)

    # gold rewrite input for qr loss
    rwrt_input = batch['rwrt_input_ids']
    # # tokens with indices set to -100 are ignored (masked)
    rwrt_input[rwrt_input == tokenizer.pad_token_id] = -100
    rwrt_input = rwrt_input.to(device)
    rwrt_attention = batch['rwrt_attention_mask'].to(device) # b, 384

    # passage input
    psg_input = batch['psg_input_ids'].to(device)
    # need to add sep token at the begining
    # roll by 1 and add column of 1s
    psg_input = torch.roll(psg_input, 1, 1)
    psg_input[:, 0] = 1

    # answer input
    ans_input = batch['ans_input_ids']
    # # tokens with indices set to -100 are ignored (masked)
    ans_input[ans_input == tokenizer.pad_token_id] = -100
    ans_input = ans_input.to(device)

    # feed context+question input and rewrite label to qr model
    qr_output = qr_model(input_ids=ctx_input, attention_mask=ctx_attention, labels=rwrt_input)

    # logits to be sampled from
    logits = qr_output.logits

    # qr loss
    qr_loss = qr_output.loss

    # gumbel softmax on the logits
    # slice upto actual vocabulary sizegumbel_softmax
    gumbel_output = F.gumbel_softmax(logits, tau=tau, hard=True)[..., :act_vocab_size]
    # print(gumbel_output.shape) # b, 384, 32100

    # normalized y cordinates for the grid. we need to select the coordinate corresponding to the vector in the vocab as output by gumbel softmax
    norm_ycord = torch.linspace(-1, 1, act_vocab_size).to(device) 
    norm_xcord = torch.linspace(-1, 1, embed_dim).to(device)	# normalized x coordinates. we need the entire vector so we will use all the coordinates

    word_embeddings = rc_model.get_input_embeddings().weight[:act_vocab_size, :]  # 32100, 768

    pad_embedding = word_embeddings[0] # embedding of <pad> token we need for masking. assuming the first embedding corresponds to the pad token
    # convert mask to float from long

    embeddings = word_embeddings.view(1, 1, act_vocab_size, -1)  # 1, 1, 32100, 768

    embeddings = embeddings.repeat(gumbel_output.shape[0], 1, 1, 1)  # b, 1, 32100, 768  repeating embeddigs batch number of times

    embedding_list = []
    
    for i in range(max_length):
        gumbeli = gumbel_output[:, i, :]  # ith token in the sequence
        gumbeli = gumbeli.view(gumbeli.shape[0], 1, -1)  # grid

        # getting normalized y coord  # 2, 1, 32100
        gumbeli = torch.mul(gumbeli, norm_ycord)
        nonz_ids = torch.nonzero(gumbeli)[:, -1] # non zero elements in each example of the batch. corresponds to the normalized id chosen by gumbel softmax
        #print(nonz_ids)

        tensor_list = []
        for j in range(len(nonz_ids)):
            gumbeli[j, :, :] = gumbeli[j, :, nonz_ids[j]] # set all elements to the non zero elements
            gumbeli_trunc = gumbeli[:, :, :embed_dim] # truncate to embed_dim
            tensorj = torch.cat((norm_xcord.view(1, embed_dim).T, gumbeli_trunc[j].T), dim = 1).view(1, embed_dim, 2)  # cat the normalized x coords
            tensor_list.append(tensorj)
            
        gumbeli = torch.cat(tensor_list, dim=0)
        gumbeli = gumbeli.view(gumbeli.shape[0], 1, embed_dim, 2) # b, 1, 768, 2
        token_embedding = F.grid_sample(embeddings, gumbeli, mode='nearest', padding_mode='border') # b, 1, 1, 768
        token_embedding = token_embedding.view(token_embedding.shape[0], -1)
        
        token_embedding = token_embedding.view(token_embedding.shape[0], 1, -1)
        embedding_list.append(token_embedding)
        
   
    # concat, mask and replace masked embeddings with embeddings of <pad>`

    inputs_embeds = torch.cat(embedding_list, dim=1)
    
    rwrt_attention_f = rwrt_attention.float()  # b, 384
    
    # mask rc input with attention mask
    mask = rwrt_attention_f.view(rwrt_attention_f.shape[0], -1, 1) @ (torch.ones(1, embed_dim)).to(device)
    inputs_embeds = torch.mul(inputs_embeds, mask)

    #print(inputs_embeds)
    #print(psg_input)
    #print(inputs_embeds.shape)
    #print(psg_input.shape)
    

    #inputs_embeds[inputs_embeds.sum(dim=2)==0] = pad_embedding

    # need to add passages
    # test word and positional embedding
   
    
    

    # use to one hot samples (straight through trick) to get vocab ids using dummy vocab
    #rc_input = gumbel_output@dummy_vocab
    #rc_input = rc_input.to(device)

    #del gumbel_output, qr_output, logits, ctx_input, ctx_attention, rwrt_input

    # mask rc input ids with attention mask
    #rc_input = torch.mul(rc_input, rwrt_attention)

    # flip the rewrite attention mask, replace 1s with 0s and vice versa
    # now the 1s represent the 'free space' in the rc_input tensor to fit the passages
    flipped_rwrt_mask = torch.fliplr(rwrt_attention)
    flipped_mask = flipped_rwrt_mask.clone()
    flipped_mask[flipped_rwrt_mask == 0] = 1
    flipped_mask[flipped_rwrt_mask == 1] = 0
    # mask passage to extract ids that can fit in the rc_input tensor
    extr_psg = torch.mul(flipped_mask, psg_input)
    # find the shifts for each row of extr_psg
    # this is equal to the number of 1s in each row of rwrt_attention
    # reshape to column vector as required by the custom gather function
    shifts = (rwrt_attention == 1).sum(dim=1).reshape(-1, 1)
    # roll each row by the amount occupied by rc_input in that row
    trunc_psg = roll_by_gather(extr_psg, 1, shifts)
    #print(trunc_psg)
    #print(trunc_psg.shape)

    trunc_psg = trunc_psg.view(trunc_psg.shape[0], -1, 1)
    #print(trunc_psg.shape)
    trunc_psg = trunc_psg.repeat(1, 1, embed_dim)

    trunc_psg = trunc_psg.float()
    
    # keep front zeros, replace end zeros with pad embedding

    for i in range(trunc_psg.shape[0]):
        flag = False
        for j in range(max_length):
            idx = trunc_psg[i][j][0].long()
            if idx == 0 and flag == False: continue
            flag = True
            #print(trunc_psg[i][j].shape)
            #print(embeddings[idx].shape)
            trunc_psg[i][j] = word_embeddings[idx]
       
    #print(trunc_psg)
    #print(trunc_psg.shape) 
    #print(pad_embedding)

    inputs_embeds = torch.add(inputs_embeds, trunc_psg)
    #print(inputs_embeds)
    #print(inputs_embeds.shape)
    #print(inputs_embeds.requires_grad)

    #rc_input = gumbel_output@dummy_vocab
    # add to get rwrt + psg as rc_input
    #rc_input = torch.add(rc_input, trunc_psg)
    # create attention mask
    #rc_attention = rc_input.clone()
    #rc_attention[rc_input != 0] = 1

    #del flipped_rwrt_mask, flipped_mask, extr_psg, shifts, trunc_psg, psg_input

    rc_loss = rc_model(inputs_embeds=inputs_embeds, labels=ans_input).loss

    #del ans_input, rc_input, rc_attention

    return qr_loss, rc_loss


def valid_loss():

    qr_epoch_loss = 0
    rc_epoch_loss = 0
    idx = 0

    for batch in test_loader:

        qr_loss, rc_loss = forward(batch)

        qr_epoch_loss += qr_loss.item()
        rc_epoch_loss += rc_loss.item()

        #del ans_input, rc_input, rc_attention
        del qr_loss, rc_loss

        idx += 1

    print('Valid loss : {}, {}'.format(qr_epoch_loss/idx, rc_epoch_loss/idx))


for epoch in range(1, num_epochs+1):

    qr_epoch_loss = 0
    rc_epoch_loss = 0

    idx = 1

    for batch in train_loader:

        qr_loss, rc_loss = forward(batch)
        #total_loss = sum([qr_loss, rc_loss])
        qr_epoch_loss += qr_loss.item()
        rc_epoch_loss += rc_loss.item()

        # total_loss.backward()
        rc_loss.backward()

        print('batch : {}'.format(idx))

        if idx % 1000 == 0:
            print('epoch {}, batch {}'.format(epoch, idx))

            """for name, param in rc_model.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)"""

        idx += 1


        optim.step()
        optim.zero_grad()



        
