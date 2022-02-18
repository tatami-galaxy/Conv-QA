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
from typing import List
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
    weight_decay : float = 0.0
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

    # dataset
    processed_dataset_dir : str = field(init=False)
    processed_dataset_format : List = field(default_factory = lambda: ['ctx_input_ids', 'rwrt_input_ids', 'psg_input_ids',
            'ans_input_ids', 'ctx_attention_mask', 'rwrt_attention_mask', 'psg_attention_mask'])

    # add methods to init dataclass attributes here
    def __post_init__(self):

        self.root = self.get_root_dir()
        self.pretrained_model = self.root + '/models/pretrained_models/t5-base'
        self.qr_finetuned = self.root + '/models/finetuned_weights/qr_gen4.pth'
        self.rc_finetuned = self.root + '/models/finetuned_weights/rc_gen5.pth'
        self.processed_dataset_dir = self.root +'/data/processed/dataset/'
        

    def get_root_dir(self):
        root = abspath(__file__)
        while root.split('/')[-1] != 'conv-qa':
            root = dirname(root)
        return root

         

class End2End(nn.Module):

    def __init__(self, options):  
        super().__init__()        

        # load T5 models
        self.qr_model = T5ForConditionalGeneration.from_pretrained(options.pretrained_model)
        self.rc_model = T5ForConditionalGeneration.from_pretrained(options.pretrained_model)



    def load_weights(self, device):

        # load finetuned weights
        self.qr_model.load_state_dict(torch.load(options.qr_finetuned, map_location=device))
        self.rc_model.load_state_dict(torch.load(options.rc_finetuned, map_location=device))  


    
    def forward(self, batch, options, device):

        # context + question input
        ctx_input = batch['ctx_input_ids'].to(device)  # QR input
        ctx_attention = batch['ctx_attention_mask'].to(device)

        # gold rewrite input for qr loss
        rwrt_input = batch['rwrt_input_ids']
        # tokens with indices set to -100 are ignored (masked)
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
        qr_output = self.qr_model(input_ids=ctx_input, attention_mask=ctx_attention, labels=rwrt_input, output_hidden_states=True)

        print(qr_output.decoder_hidden_states[0].is_leaf)

        # logits to be sampled from
        logits = qr_output.logits


        # qr loss
        qr_loss = qr_output.loss

        # gumbel softmax on the logits
        # slice upto actual vocabulary sizegumbel_softmax
        gumbel_output = F.gumbel_softmax(logits, tau=options.tau, hard=True)[..., :options.act_vocab_size]

        # print(gumbel_output.shape) # b, 384, 32100

        # normalized y cordinates for the grid
        # we need to select the coordinate corresponding to the vector in the vocab as output by gumbel softmax
        norm_ycord = torch.linspace(-1, 1, options.act_vocab_size).to(device) 
	# normalized x coordinates. we need the entire vector so we will use all the coordinates
        norm_xcord = torch.linspace(-1, 1, options.embed_dim).to(device)

        # T5 input embeddings
        word_embeddings = self.rc_model.get_input_embeddings().weight[:options.act_vocab_size, :]  # 32100, 768

        # embedding of <pad> token we need for masking. assuming the first embedding corresponds to the pad token
        pad_embedding = word_embeddings[0] 

        # reshape embeddings for input to grid_sample
        embeddings = word_embeddings.view(1, 1, options.act_vocab_size, -1)  # 1, 1, 32100, 768
        embeddings = embeddings.repeat(gumbel_output.shape[0], 1, 1, 1)  # b, 1, 32100, 768  repeating embeddigs batch number of times

        # list to store the embeddings per max_length position
        embedding_list = []
    
        for i in range(options.max_length):
            gumbeli = gumbel_output[:, i, :]  # ith token in the sequence
            gumbeli = gumbeli.view(gumbeli.shape[0], 1, -1)  # reshaping to make grid

            # getting normalized y coord  # b, 1, 32100
            gumbeli = torch.mul(gumbeli, norm_ycord)  # replaces the 1.0 with the normalized y coordinate
            # non zero elements in each example of the batch. corresponds to the normalized id chosen by gumbel softmax
            nonz_ids = torch.nonzero(gumbeli)[:, -1]
            #print(nonz_ids)

            # list to hold reshaped gumbeli containing the grid to extract embeddings
            tensor_list = []
            for j in range(len(nonz_ids)):
                gumbeli[j, :, :] = gumbeli[j, :, nonz_ids[j]] # set all elements to the non zero elements
                gumbeli_trunc = gumbeli[:, :, :options.embed_dim] # truncate to embed_dim

                # cat the normalized x coordinates
                tensorj = torch.cat((norm_xcord.view(1, options.embed_dim).T, gumbeli_trunc[j].T), dim = 1).view(1, options.embed_dim, 2)
                tensor_list.append(tensorj)
            
            gumbeli = torch.cat(tensor_list, dim=0) # reshaped gumbeli with grid
            gumbeli = gumbeli.view(gumbeli.shape[0], 1, options.embed_dim, 2) # b, 1, 768, 2

            token_embedding = F.grid_sample(embeddings, gumbeli, mode='nearest', padding_mode='border') # b, 1, 1, 768
            token_embedding = token_embedding.view(token_embedding.shape[0], -1)
        
            token_embedding = token_embedding.view(token_embedding.shape[0], 1, -1)
            embedding_list.append(token_embedding)
        
   
        # concat embeddings for max_length positions for batch
        inputs_embeds = torch.cat(embedding_list, dim=1)
    
        # cast rewrite attention mask to float
        rwrt_attention_f = rwrt_attention.float()  # b, 384
    
        # mask rc input (inputs_embeds) with attention mask
        # masked positions are replaced with 0.0 vectors
        mask = rwrt_attention_f.view(rwrt_attention_f.shape[0], -1, 1) @ (torch.ones(1, options.embed_dim)).to(device)  # reshape mask
        inputs_embeds = torch.mul(inputs_embeds, mask)

        #print(inputs_embeds)
        #print(psg_input)
        #print(inputs_embeds.shape)
        #print(psg_input.shape)
        #inputs_embeds[inputs_embeds.sum(dim=2)==0] = pad_embedding

        # now we need to fit the passage embeddings after the rewrite embeddings
        # flip the original rewrite attention mask, replace 1s with 0s and vice versa
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
        trunc_psg = roll_by_gather(extr_psg, 1, shifts, device)

        #print(trunc_psg)
        #print(trunc_psg.shape)

        # reshape and repeat values for changing into embeddings afterwards
        trunc_psg = trunc_psg.view(trunc_psg.shape[0], -1, 1)
        trunc_psg = trunc_psg.repeat(1, 1, options.embed_dim)
        # cast to float
        trunc_psg = trunc_psg.float()
    
        # need to keep front zeros, since they will be replaced by the rewrite embeddings
        # replace end zeros with pad embedding
        for i in range(trunc_psg.shape[0]):
            flag = False
            for j in range(options.max_length):
                idx = trunc_psg[i][j][0].long()
                if idx == 0 and flag == False: continue
                flag = True
                #print(trunc_psg[i][j].shape)
                #print(embeddings[idx].shape)
                trunc_psg[i][j] = word_embeddings[idx]
       
        #print(trunc_psg)
        #print(trunc_psg.shape) 
        #print(pad_embedding)

        # add inputs_embeds and masked passage embeddings
        inputs_embeds = torch.add(inputs_embeds, trunc_psg)

        #print(inputs_embeds)
        #print(inputs_embeds.shape)
        #print(inputs_embeds.requires_grad)

        rc_loss = self.rc_model(inputs_embeds=inputs_embeds, labels=ans_input).loss

        #rc_loss.backward(retain_graph=True)        
        #print(inputs_embeds.grad)



        return qr_loss, rc_loss



if __name__ == '__main__':

    device = torch.device('cpu')

    # hyperparameters and other options
    options = Options()

    # end to end model
    e2epipe = End2End(options)
    e2epipe.to(device) 
    e2epipe.load_weights(device)  # finetuned weights
    e2epipe.train()

    # tokenizer (need to save)
    tokenizer = T5Tokenizer.from_pretrained(options.pretrained_model_name)

    # optimizer
    optim = Adafactor(
            e2epipe.parameters(),
            lr = options.lr,
            eps = options.eps,
            clip_threshold = options.clip_threshold,
            decay_rate = options.decay_rate,
            beta1 = options.beta1,
            weight_decay = options.weight_decay,
            relative_step= options.relative_step,
            scale_parameter = options.scale_parameter,
            warmup_init = options.warmup_init)

    # dataset

    dataset = load_from_disk(options.processed_dataset_dir)
    dataset.set_format(type='torch', columns = options.processed_dataset_format,)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=options.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=options.batch_size)

    # train loop
    for epoch in range(1, options.num_epochs + 1):

        idx = 1

        for batch in train_loader:

            qr_loss, rc_loss = e2epipe(batch, options, device)  

            #if idx % 100 == 0:
            print('epoch {}, batch {}'.format(epoch, idx))

            idx += 1

            #optim.zero_grad()
            #rc_loss.backward()

            #for name, param in e2epipe.qr_model.named_parameters():
                #if param.requires_grad: print(name, param.grad)

            #optim.step()




    


        
