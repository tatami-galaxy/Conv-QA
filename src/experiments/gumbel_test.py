import json
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F

# qr_model.save_pretrained('~/Documents/conv-qa/models/pretrained_models/t5-base')

device = torch.device('cpu')
# device = torch.device('gpu')

max_length = 384
batch_size = 4
embed_dim = 768

pretrained_model = 't5-base'

tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
qr_model = T5ForConditionalGeneration.from_pretrained('/home/ujan/Documents/conv-qa/models/pretrained_models/t5-base')
rc_model = T5ForConditionalGeneration.from_pretrained('/home/ujan/Documents/conv-qa/models/pretrained_models/t5-base')

qr_model.load_state_dict(torch.load('/home/ujan/Documents/conv-qa/models/finetuned_weights/qr_gen4.pth'))
rc_model.load_state_dict(torch.load('/home/ujan/Documents/conv-qa/models/finetuned_weights/rc_gen5.pth'))

act_vocab_size = len(tokenizer.get_vocab())
num_epochs = 2

optim = Adafactor(
    # list(qr_model.parameters())+list(rc_model.parameters()),
    qr_model.parameters(),
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

dataset = load_from_disk(
    '/home/ujan/Documents/conv-qa/data/processed/dataset/')
dataset.set_format(type='torch', columns=['ctx_input_ids', 'rwrt_input_ids', 'psg_input_ids',
                   'ans_input_ids', 'ctx_attention_mask', 'rwrt_attention_mask', 'psg_attention_mask'],)

train_loader = torch.utils.data.DataLoader(
    dataset['train'], batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
    dataset['test'], batch_size=batch_size)


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
    qr_output = qr_model(input_ids=ctx_input,
                         attention_mask=ctx_attention, labels=rwrt_input)

    # logits to be sampled from
    logits = qr_output.logits

    # qr loss
    qr_loss = qr_output.loss

    # gumbel softmax on the logits
    # slice upto actual vocabulary sizegumbel_softmax
    gumbel_output = F.gumbel_softmax(logits, tau=1, hard=True)[..., :act_vocab_size]
    # print(gumbel_output.shape) # b, 384, 32100

    norm_ycord = torch.linspace(-1, 1, act_vocab_size).to(device) # normalized y cordinates for the grid. we need to select the coordinate corresponding to the vector in the vocab as output by gumbel softmax
    norm_xcord = torch.linspace(-1, 1, embed_dim).to(device)	# normalized x coordinates. we need the entire vector so we will use all the coordinates

    embeddings = rc_model.get_input_embeddings().weight[:act_vocab_size, :]  # 32100, 768
    #print(embeddings[410])
    embeddings = embeddings.view(1, 1, act_vocab_size, -1)  # 1, 1, 32100, 768

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
        
        embedding_list.append(token_embedding)
        
    print(len(embedding_list))
    print(embedding_list[0].shape)

    # concat, mask and replace masked embeddings with embeddings of <pad>

    # use to one hot samples (straight through trick) to get vocab ids using dummy vocab
    rc_input = gumbel_output@dummy_vocab
    rc_input = rc_input.to(device)

    del gumbel_output, qr_output, logits, ctx_input, ctx_attention, rwrt_input

    # mask rc input ids with attention mask
    rc_input = torch.mul(rc_input, rwrt_attention)
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
    # add to get rwrt + psg as rc_input
    rc_input = torch.add(rc_input, trunc_psg)
    # create attention mask
    rc_attention = rc_input.clone()
    rc_attention[rc_input != 0] = 1

    del flipped_rwrt_mask, flipped_mask, extr_psg, shifts, trunc_psg, psg_input

    rc_loss = rc_model(input_ids=rc_input,
                       attention_mask=rc_attention, labels=ans_input).loss

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

        if idx % 100 == 0:
            print('epoch {}, batch {}'.format(epoch, idx))

            for name, param in rc_model.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)

        optim.step()
        optim.zero_grad()

        break

    break
