import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
#from torchviz import make_dot

@dataclass
class Options:  # class for storing hyperparameters and other options

    # model hyperparameters
    max_length : int = 384  # use interim data if changed 
    batch_size : int = 8
    embed_dim : int = 768  # typical base model embedding dimension
    pretrained_t5_model_name : str = 't5-base'
    pretrained_para_model_name : str = 't5-base'
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

    # tokenizers
    t5_tokenizer : any = field(init=False)
    para_tokenizer : any = field(init=False)

    # directory names
    pretrained_t5_model_dir = '/models/pretrained_models/t5-base'
    pretrained_para_model_dir = '/models/pretrained_models/t5-base'
    rc_finetuned_dir = '/models/finetuned_weights/new_rc_gen5.pth'
    para_finetuned_dir = '/models/finetuned_weights/para_5.pth'
    t5_tokenizer_dir_name = '/models/pretrained_models/t5-tokenizer'
    processed_dataset_dir_name = '/data/processed/dataset/'
    interim_dataset_dir_name = '/data/interim/'

    # directories
    root : str = field(init=False)
    pretrained_t5_model : str = field(init=False)
    rc_finetuned : str = field(init=False)
    para_finetuned : str = field(init=False)
    t5_tokenizer_dir : str = field(init=False)

    # dataset
    processed_dataset_dir : str = field(init=False)
    interim_dataset_dir : str = field(init=False)
    processed_dataset_format : List = field(default_factory = lambda: ['ctx_input_ids', 'rwrt_input_ids', 'psg_input_ids',
            'ans_input_ids', 'ctx_attention_mask', 'rwrt_attention_mask', 'psg_attention_mask'])


    # add methods to init dataclass attributes here
    def __post_init__(self):

        # root
        self.root = self.get_root_dir()

        # models
        self.pretrained_t5_model = self.root + self.pretrained_t5_model_dir

        # finetuned weights
        self.rc_finetuned = self.root + self.rc_finetuned_dir
        self.para_finetuned = self.root + self.para_finetuned_dir

        # tokenizers
        self.t5_tokenizer_dir = self.root + self.t5_tokenizer_dir_name
        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_tokenizer_dir)

        # datasets
        self.processed_dataset_dir = self.root + self.processed_dataset_dir_name
        self.interim_dataset_dir = self.root + self.interim_dataset_dir_name
        

        

    def get_root_dir(self):
        root = abspath(__file__)
        while root.split('/')[-1] != 'conv-qa':
            root = dirname(root)
        return root

         

class End2End(nn.Module):

    def __init__(self, options):  
        super().__init__()        

        # load models
        self.rc_model = T5ForConditionalGeneration.from_pretrained(options.pretrained_t5_model)
        self.para_model = T5ForConditionalGeneration.from_pretrained(options.pretrained_t5_model)



    def load_weights(self, device):

        # load finetuned weights
        self.rc_model.load_state_dict(torch.load(options.rc_finetuned, map_location=device))  
        self.para_model.load_state_dict(torch.load(options.rc_finetuned, map_location=device))  


    def save_models(self, options, epoch):

        torch.save(self.rc_model.state_dict(), options.root+'/models/finetuned_weights/e2e_para_rc'+str(epoch)+'.pth')
        torch.save(self.para_model.state_dict(), options.root+'/models/finetuned_weights/e2e_para'+str(epoch)+'.pth')

    
    def forward(self, batch, options, device):

        # gold rewrite 
        rwrt_input = batch['rwrt_input_ids']
        rwrt_input = rwrt_input.to(device)
        rwrt_label = batch['rwrt_input_ids']
        # tokens with indices set to -100 are ignored (masked)
        rwrt_label[rwrt_label == options.t5_tokenizer.pad_token_id] = -100
        rwrt_label = rwrt_label.to(device)
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
        ans_input[ans_input == options.t5_tokenizer.pad_token_id] = -100
        ans_input = ans_input.to(device)

        # feed rewrite to para model
        para_output = self.para_model(input_ids=rwrt_input, attention_mask=rwrt_attention, labels=rwrt_label, output_hidden_states=True)  # max length 256


        # logits to be sampled from
        logits = para_output.logits
        #logits.retain_grad()

        # gumbel softmax on the logits
        # slice upto actual vocabulary size
        gumbel_output = F.gumbel_softmax(logits, tau=options.tau, hard=True)[..., :options.act_vocab_size]

        # print(gumbel_output.shape) # b, 384, 32100

        # normalized y cordinates for the grid
        # we need to select the coordinate corresponding to the vector in the vocab as output by gumbel softmax
        #norm_ycord = torch.linspace(-1, 1, options.act_vocab_size).to(device) 
        # normalized x coordinates. we need the entire vector so we will use all the coordinates
        #norm_xcord = torch.linspace(-1, 1, options.embed_dim).to(device)

        # T5 input embeddings
        word_embeddings = self.rc_model.get_input_embeddings().weight[:options.act_vocab_size, :]  # 32100, 768

        # embedding of <pad> token we need for masking. assuming the first embedding corresponds to the pad token
        pad_embedding = word_embeddings[0] 

        #
 
        batch_list = []
        dummy = torch.ones(options.act_vocab_size).to(device)

        for i in range(gumbel_output.shape[0]):

            embedding_list = []

            for j in range(options.max_length):       
                ind = gumbel_output[i][j].expand(options.embed_dim, -1).T
                #ind = gumbel_output[i][j].repeat(options.embed_dim, 1).T
                embedding_list.append(torch.mul(ind, word_embeddings).T @ dummy)

            batch_list.append(torch.stack(embedding_list, dim=0))

        inputs_embeds = torch.stack(batch_list, dim=0)  
 
        # concat embeddings for max_length positions for batch
        #inputs_embeds = torch.cat(embedding_list, dim=1)

    
        # cast rewrite attention mask to float
        rwrt_attention_f = rwrt_attention.float()  # b, 384
    
        # mask rc input (inputs_embeds) with attention mask
        # masked positions are replaced with 0.0 vectors
        mask = rwrt_attention_f.view(rwrt_attention_f.shape[0], -1, 1) @ (torch.ones(1, options.embed_dim)).to(device)  # reshape mask

        #print(inputs_embeds)
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

        rc_loss = self.rc_model(inputs_embeds=inputs_embeds, labels=ans_input).loss

        return rc_loss


class DataClass:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_csv(self, f, output):

        answers = []
        rewrites = []
        passages = []

        filepath = self.data_dir+f

        with open(filepath) as fl:
            data = json.load(fl)
      
        for d in data:
            answers.append(d['answer'])
            rewrites.append(d['rewrite'])
            passages.append(d['passage'])

        data = {'answer':answers, 'passage':passages, 'rewrite':rewrites}
        df = pd.DataFrame(data)
        df.to_csv(output, index=False)



def tokenize_dataset(batch, options):

    passages = options.para_tokenizer(batch['passage'], padding='max_length',
        truncation=True, max_length=options.max_length, add_special_tokens=True)

    rewrites = options.para_tokenizer(batch['rewrite'], padding='max_length',
        truncation=True, max_length=options.max_length, add_special_tokens=True)

    answers = options.para_tokenizer(batch['answer'], padding='max_length',
        truncation=True, max_length=options.max_length, add_special_tokens=True)

    labels = options.tokenizer(batch['label'], padding='max_length', truncation=True,
            max_length=options.max_length, add_special_tokens=True)

    batch['psg_input_ids'] = passages.input_ids
    batch['rwrt_input_ids'] = rewrites.input_ids
    batch['ans_input_ids']  = answers.input_ids
    batch['lbl_input_ids']  = labels.input_ids
    batch['psg_attention_mask'] = passages.attention_mask
    batch['rwrt_attention_mask'] = rewrites.attention_mask
    batch['lbl_attention_mask'] = labels.attention_mask

    return batch


# handle examples with no answers
def no_ans(x):
    if isinstance(x['answer'], str): return x
    x['answer'] = 'no_ans'
    return x



if __name__ == '__main__':

    device = torch.device('cpu')

    # hyperparameters and other options
    options = Options()

    # load dataset
    qrecc = load_from_disk()##

    # no answers
    qrecc = qrecc.map(no_ans)

    # tokenizing
    dataset = qrecc.map(tokenize_dataset, fn_kwargs={'options': options}, batch_size = options.batch_size,
        batched=True, remove_columns=['passage', 'answer', 'rewrite', 'label'])

    dataset.set_format(
        type='torch', columns=['rwrt_input_ids', 'psg_input_ids', 'ans_input_ids', 'psg_attention_mask', 'rwrt_attention_mask',
            'lbl_input_ids', 'lbl_attention_mask'],)


    # end to end model
    e2epipe = End2End(options)
    e2epipe.to(device) 
    #e2epipe.load_weights(device)  # finetuned weights
    e2epipe.train()

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

    # dataloaders
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=options.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=options.batch_size)

    print('Number of batches : {}'.format(len(train_loader)))

    print('Start training')

    # train loop
    for epoch in range(1, options.num_epochs + 1):

        idx = 1

        rc_epoch_loss = 0

        for batch in train_loader:

            rc_loss = e2epipe(batch, options, device)  

            rc_epoch_loss += rc_loss.item()


            if idx % 500 == 0:
                print('epoch {}, batch {}'.format(epoch, idx))
 
            idx += 1

            optim.zero_grad()
            rc_loss.backward()

            #for name, param in e2epipe.qr_model.named_parameters():
                #if param.requires_grad: print(name, param.grad)

            optim.step()


        print('Train loss : {}'.format(rc_epoch_loss/len(train_loader)))

        e2epipe.eval()

        # valid loop
        rc_valid_loss = 0

        idx = 0

        for batch in test_loader:

            rc_loss = e2epipe(batch, options, device)
            rc_valid_loss += rc_loss.item()

            idx += 1


        print('Valid loss : {}'.format(rc_valid_loss/idx))

        print('\n')

        e2epipe.train()
        #e2epipe.save_models(options, epoch)
        #print('Model saved')



 





    


       
