import os
from os.path import dirname, abspath
import sys

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'conv-qa':
    root = dirname(root)

# append to path for loading class
sys.path.append(root+'/src/joint_training/')

from datasets import load_dataset, load_metric, load_from_disk
from transformers import T5Tokenizer
from modeling_flax_t5 import FlaxT5ForConditionalGeneration
from transformers import set_seed
import datasets
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
import evaluate
import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
import flax.linen as nn
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from typing import Callable, Optional
import math
import nltk
nltk.download('punkt') # needed for rouge score
import time
from functools import partial
import collections
from typing import List


# hyperparameters
max_length = 256  # can also have different max_length for train, eval, generate
num_beams = 1  # 1 -> no beam search
per_device_train_batch_size = 16
per_device_eval_batch_size= 16
seed = 42 # 37, 52
num_train_epochs = 5
warmup_steps = 1000
learning_rate = 1e-5
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
weight_decay = 0.0
label_smoothing_factor = 0.0


# directories
output_dir = root+'/models/'
qr_output_dir = root+'/models/qr/'
rc_output_dir = root+'/models/rc/'
data_dir = root+'/data/interim/qrecc/'
processed_data_dir = root+'/data/processed/qrecc/'
# models
# same for both qr and rc as of now
model_name = 't5-base'

# Metrics #
qr_metric = evaluate.load("rouge")
# rc metric is F1


# data loader
def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


# in Flax, for seq2seq models we need to pass `decoder_input_ids`
# as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
# `shift_tokens_right` function
# copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = qr_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# F1 metric
def compute_f1_from_tokens(gold_toks: List[str], pred_toks: List[str]) -> float:
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # if either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# learning rate function
def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


# we use Optax's "masking" functionality to not apply weight decay
# to bias and LayerNorm scale parameters. decay_mask_fn returns a
# mask boolean with the same structure as the parameters.
# the mask is True for parameters that should be decayed.
def decay_mask_fn(params):
    # model parameters -> pytrees

    # flatten a nested dictionary
    # the nested keys are flattened to a tuple
    #xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    #flat_xs = flatten_dict(xs)
    #print(flat_xs)
    # {
    #   ('foo',): 1,
    #   ('bar', 'a'): 2,
    # }

    # mask* -> a tree with same structure as (or a prefix of) the params PyTree,
    # or a Callable that returns such a pytree given the params/updates.
    # the leaves should be booleans, True for leaves/subtrees you want to apply the weight decay to,
    # and False for those you want to skip.
    # note that the Adam gradient transformations are applied to all parameters.

    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = set(
        [
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        ]
    )
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}

    # flat_xs = {
    #  ('foo',): 1,
    #  ('bar', 'a'): 2,
    #}
    # xs = unflatten_dict(flat_xs)
    # print(xs)
    # {
    #   'foo': 1
    #   'bar': {'a': 2}
    # }

    return traverse_util.unflatten_dict(flat_mask)


# setup train state

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng))



# label smoothed cross entropy
def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
    """
    The label smoothing implementation is adapted from Flax's official example:
    https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
    """
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing_factor
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
    soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)
    loss = optax.softmax_cross_entropy(logits, soft_labels)
    loss = loss - normalizing_constant

    # ignore padded tokens from loss
    loss = loss * padding_mask
    loss = loss.sum()
    num_labels = padding_mask.sum()
    return loss, num_labels


# define gradient update step fn
def train_step(qr_state, rc_state, batch, label_smoothing_factor=0.0):
    dropout_rng, qr_dropout_rng = jax.random.split(qr_state.dropout_rng)
    dropout_rng, rc_dropout_rng = jax.random.split(rc_state.dropout_rng)

    def compute_loss_qr(params):
        labels = batch.pop("qr_labels")
        # params for FLaxT5PreTrainedModel __call__
        logits = qr_state.apply_fn(
            input_ids=batch["qr_input_ids"],
            attention_mask=batch["qr_attention_mask"],
            decoder_input_ids=batch["qr_decoder_input_ids"],
            decoder_attention_mask=batch["qr_decoder_attention_mask"],
            params=params,
            dropout_rng=qr_dropout_rng,
            train=True)[0]
        loss, num_labels = loss_fn(logits, labels, batch["qr_decoder_attention_mask"], label_smoothing_factor)
        return loss, num_labels  # therefore has_aux=True

    # also update qr here
    def compute_loss_rc(params, rc_input):  ##### fix #####
        labels = batch.pop("rc_labels")
        # params for FLaxT5PreTrainedModel __call__
        logits = rc_state.apply_fn(
            input_ids=rc_input,
            attention_mask=batch["rc_attention_mask"],
            decoder_input_ids=batch["rc_decoder_input_ids"],
            decoder_attention_mask=batch["rc_decoder_attention_mask"],
            params=params,
            dropout_rng=rc_dropout_rng,
            train=True)[0]
        loss, num_labels = loss_fn(logits, labels, batch["rc_decoder_attention_mask"], label_smoothing_factor)
        return loss, num_labels  # therefore has_aux=True

    # compute rc loss and qr, rc grads
    # compute rc loss and rc grads
    # update

    ## value_and_grad ##
    # argnums (Union[int, Sequence[int]])
    # pptional, integer or sequence of integers.
    # specifies which positional argument(s) to differentiate with respect to (default 0).

    # value_and grad returns a ** Function ** with the same arguments as fun
    # that evaluates both fun and the gradient of fun and returns them as a pair (a two-element tuple).
    # if argnums is an integer then the gradient has the same shape and type
    # as the positional argument indicated by that integer.
    # if argnums is a sequence of integers, the gradient is a tuple of values
    # with the same shapes and types as the corresponding arguments.
    # if has_aux is True then a tuple of ((value, auxiliary_data), gradient) is returned.

    # rc forward #

    #rc_input = batch["rc_input_ids"]
    #rc_grad_fn = jax.value_and_grad(compute_loss_rc, has_aux=True, argnums=[0,1])  ### fix ###

    # qr forward #
    qr_grad_fn = jax.value_and_grad(compute_loss_qr, has_aux=True)

    # has_aux (bool)
    # indicates whether fun (Function to be differentiated) returns a pair where the first element
    # is considered the output of the mathematical function to be differentiated
    # and the second element is auxiliary data. Default False

    #(loss, num_labels), grad = rc_grad_fn(rc_state.params, batch["rc_input_ids"])

    # what grads are being computed?
    # need grads wrt to the qr (string) input
    # how to copy them back to the rc graph and propagate through graph?

    # copy grads from rc to qr here


    (loss, num_labels), grad = qr_grad_fn(qr_state.params)

    num_labels = jax.lax.psum(num_labels, "batch")

    # true loss = total loss / total samples
    loss = jax.lax.psum(loss, "batch")
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    # true grad = total grad / total samples
    grad = jax.lax.psum(grad, "batch")
    grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
    new_state = qr_state.apply_gradients(grads=grad, dropout_rng=qr_dropout_rng)

    metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(qr_state.step)}
    return new_state, metrics



# define eval fn
def eval_step(params, batch, model_str, label_smoothing_factor=0.0):
    if model_str == 'qr':
        labels = batch.pop("qr_labels")
        logits = qr_model(
            input_ids=batch["qr_input_ids"],
            attention_mask=batch["qr_attention_mask"],
            decoder_input_ids=batch["qr_decoder_input_ids"],
            decoder_attention_mask=batch["qr_decoder_attention_mask"],
            params=params,
            train=False)[0]
        loss, num_labels = loss_fn(logits, labels, batch["qr_decoder_attention_mask"], label_smoothing_factor)
    elif model_str == 'rc':
        labels = batch.pop("rc_labels")
        logits = rc_model(
            input_ids=batch["rc_input_ids"],
            attention_mask=batch["rc_attention_mask"],
            decoder_input_ids=batch["rc_decoder_input_ids"],
            decoder_attention_mask=batch["rc_decoder_attention_mask"],
            params=params,
            train=False)[0]
        loss, num_labels = loss_fn(logits, labels, batch["rc_decoder_attention_mask"], label_smoothing_factor)

    num_labels = jax.lax.psum(num_labels, "batch")

    # true loss = total loss / total samples

    # compute an all-reduce sum on x over the pmapped axis axis_name (batch)
    # if x is a pytree then the result is equivalent to mapping this function to each leaf in the tree.
    # inputs of boolean dtype are converted to integers before the reduction

    loss = jax.lax.psum(loss, "batch")
    # maps a multi-input function over pytree args to produce a new pytree
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    metrics = {"loss": loss}
    return metrics



def generate_step(params, batch, model_str):
    if model_str == 'qr':
        qr_model.params = params
        output_ids = qr_model.generate(batch["qr_input_ids"], attention_mask=batch["qr_attention_mask"], **gen_kwargs)
    elif model_str == 'rc':
        rc_model.params = params
        output_ids = rc_model.generate(batch["rc_input_ids"], attention_mask=batch["rc_attention_mask"], **gen_kwargs)
    return output_ids.sequences




if __name__ == '__main__':
    
    # set seed
    set_seed(seed)
    
    # tokenizer and Model #
    print('loading tokenizer')
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)
    print('loading models')
    qr_model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)
    rc_model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)


    # Dataset #

    # check if processed data exists
    if os.path.isdir(processed_data_dir) and os.listdir(processed_data_dir):
        print('processed data found at {}'.format(processed_data_dir))
        dataset = load_from_disk(processed_data_dir)

    else:

        print('loading raw dataset')
        qrecc = load_from_disk(data_dir)
        #print(qrecc)

        # removing examples with no context
        ### maybe not?
        print('removing examples with no context')
        qrecc = qrecc.filter(lambda x: isinstance(x['context'], str) and isinstance(x['rewrite'], str))

        # tokenize dataset function
        def tokenize_dataset(batch):
            # need fixed length inputs for jitted functions
            qr_inputs = tokenizer(
                batch['context'], batch['question'], padding='max_length',
                truncation='only_first', max_length=max_length, return_tensors="np")

            rc_inputs = tokenizer(
                batch['passage'], batch['rewrite'], padding='max_length',
                truncation='only_first', max_length=max_length, return_tensors="np")
            
            qr_labels = tokenizer(
                text_target=batch['rewrite'],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",)

            rc_labels = tokenizer(
                text_target=batch['answer'],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",)

            # prepare decoder input ids by shifting tokens right
            # no 'labels' in flax model
            qr_decoder_input_ids = shift_tokens_right(
                qr_labels["input_ids"], qr_model.config.pad_token_id, qr_model.config.decoder_start_token_id)
            rc_decoder_input_ids = shift_tokens_right(
                rc_labels["input_ids"], rc_model.config.pad_token_id, rc_model.config.decoder_start_token_id)

            # combine dicts
            model_inputs = {}
            # input ids
            model_inputs["qr_input_ids"] = qr_inputs["input_ids"]
            model_inputs["rc_input_ids"] = rc_inputs["input_ids"]
            # attention mask
            model_inputs["qr_attention_mask"] = qr_inputs["attention_mask"]
            model_inputs["rc_attention_mask"] = rc_inputs["attention_mask"]
            # labels
            model_inputs["qr_labels"] = qr_labels["input_ids"]
            model_inputs["rc_labels"] = rc_labels["input_ids"]
            # decoder input ids
            model_inputs["qr_decoder_input_ids"] = np.asarray(qr_decoder_input_ids)
            model_inputs["rc_decoder_input_ids"] = np.asarray(rc_decoder_input_ids)
            # decoder attention mask
            # we need decoder_attention_mask so we can ignore pad tokens from loss
            model_inputs["qr_decoder_attention_mask"] = qr_labels["attention_mask"]
            model_inputs["rc_decoder_attention_mask"] = rc_labels["attention_mask"]

            return model_inputs

        # tokenize dataset
        # input_ids, attention_mask, labels, decoder_input_ids, decoder_attention_mask for qr and rc
        dataset = qrecc.map(
            tokenize_dataset,
            batched=True,
            remove_columns=qrecc['train'].column_names,
            desc="tokenizing dataset",)

        dataset.save_to_disk(processed_data_dir)
    


    # Training #
    # rng
    # JAX’s random functions produce pseudorandom numbers from the PRNG state, but do not change the state
    # reusing the same state will cause sadness and monotony, depriving the end user of lifegiving chaos
    # instead, we split the PRNG to get usable subkeys every time we need a new pseudorandom number
    # old key -> new key, new subkey
    # we propagate the key and make new subkeys whenever we need a new random number
    # print("old key", key)
    # key, subkey = random.split(key)
    # normal_pseudorandom = random.normal(subkey, shape=(1,))
    # print("    \---SPLIT --> new key   ", key)
    # print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
    print('jax rng')
    rng = jax.random.PRNGKey(seed)
    rng, dropout_rng = jax.random.split(rng)

    # training constants
    num_epochs = int(num_train_epochs)
    train_batch_size = int(per_device_train_batch_size) 
    per_device_eval_batch_size = int(per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(dataset['train']) // train_batch_size#
    total_train_steps = steps_per_epoch * num_epochs#

    # learning rate schedule
    print('learning rate schedule')
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(dataset['train']),
        train_batch_size,
        num_train_epochs,
        warmup_steps,
        learning_rate,)

    # adam optimizer
    print('optimizer')
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=adam_beta1,
        b2=adam_beta2,
        eps=adam_epsilon,
        weight_decay=weight_decay,
        mask=decay_mask_fn,) # mask*

    # train states
    # FLaxT5PreTrainedModel __call__ 
    # 2 states for 2 models
    qr_state = TrainState.create(
        apply_fn=qr_model.__call__, params= qr_model.params,  
        tx=adamw, dropout_rng=dropout_rng)
    rc_state = TrainState.create(
        apply_fn=rc_model.__call__, params=rc_model.params,
        tx=adamw, dropout_rng=dropout_rng)


    # define generation function for eval with generation
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


    # create parallel version of the train and eval step
    # static_broadcasted_argnums -> an int or collection of ints 
    # specifying which positional arguments to treat as static (compile-time constant)
    p_train_step = jax.pmap(partial(
        train_step,
        label_smoothing_factor=label_smoothing_factor), "batch",
        donate_argnums=(0,)
    )  # donate_argnums ?

    p_eval_step = jax.pmap(partial(
        eval_step,
        label_smoothing_factor=label_smoothing_factor), "batch",
    )

    p_generate_step = jax.pmap(
            generate_step, "batch",
            static_broadcasted_argnums=2)


    # replicate the train states on each device
    qr_state = qr_state.replicate()
    rc_state = rc_state.replicate()

    train_time = 0
    print('starting training')
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        ### training ###
        train_start = time.time()

        # create sampling rng
        rng, input_rng = jax.random.split(rng)
        qr_train_metrics = []
        rc_train_metrics = []

        # generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, dataset['train'], train_batch_size, shuffle=True)
        steps_per_epoch = len(dataset['train']) // train_batch_size
        # train
        for _ in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)
            batch = shard(batch) # for multi gpu

            qr_state, qr_train_metric = p_train_step(qr_state, rc_state, batch)

            qr_train_metrics.append(qr_train_metric)
            rc_train_metrics.append(rc_train_metric)

        train_time += time.time() - train_start

        qr_train_metric = unreplicate(qr_train_metric)
        rc_train_metric = unreplicate(rc_train_metric)

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | QR loss: {qr_train_metric['loss']}, QR Learning Rate:"
            f" {qr_train_metric['learning_rate']})"
            f" | RC loss: {rc_train_metric['loss']}, RC Learning Rate:"
            f" {rc_train_metric['learning_rate']})"
        )

        ### evaluation ###
        qr_eval_metrics = []
        qr_eval_preds = []
        qr_eval_labels = []

        rc_eval_metrics = []
        rc_eval_preds = []
        rc_eval_labels = []

        f1_score = 0

        eval_loader = data_loader(input_rng, dataset['valid'], eval_batch_size, drop_last=False)
        eval_steps = math.ceil(len(dataset['valid']) / eval_batch_size)
        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
            # model forward
            batch = next(eval_loader)
            qr_labels = batch["qr_labels"]
            rc_labels = batch["rc_labels"]

            qr_metrics = pad_shard_unpad(p_eval_step, static_argnums=(0, 2), static_return=True)(
                qr_state.params, batch, 'qr', min_device_batch=per_device_eval_batch_size
            )
            rc_metrics = pad_shard_unpad(p_eval_step, static_argnums=(0, 2), static_return=True)(
                rc_state.params, batch, 'rc', min_device_batch=per_device_eval_batch_size
            )
            qr_eval_metrics.append(qr_metrics)
            rc_eval_metrics.append(rc_metrics)

            # generation
            qr_generated_ids = pad_shard_unpad(p_generate_step, static_argnums=(0, 2))(qr_state.params, batch, 'qr')
            rc_generated_ids = pad_shard_unpad(p_generate_step, static_argnums=(0, 2))(rc_state.params, batch, 'rc')

            qr_eval_preds.extend(jax.device_get(qr_generated_ids.reshape(-1, gen_kwargs["max_length"])))
            rc_eval_preds.extend(jax.device_get(rc_generated_ids.reshape(-1, gen_kwargs["max_length"])))

            qr_eval_labels.extend(qr_labels)
            rc_eval_labels.extend(rc_labels)

        # normalize eval metrics
        qr_eval_metrics = get_metrics(qr_eval_metrics)
        qr_eval_metrics = jax.tree_util.tree_map(jnp.mean, qr_eval_metrics)

        rc_eval_metrics = get_metrics(rc_eval_metrics)
        rc_eval_metrics = jax.tree_util.tree_map(jnp.mean, rc_eval_metrics)

        # compute ROUGE metrics
        rouge_desc = ""
        qr_rouge_metrics = compute_metrics(qr_eval_preds, qr_eval_labels)
        qr_eval_metrics.update(qr_rouge_metrics)
        rouge_desc = " ".join([f"QR Eval {key}: {value} |" for key, value in qr_rouge_metrics.items()])

        # compute F1 metrics
        # decode to strings
        # eval_preds (and labels) -> list of all preds, each a numpy array of token_ids of max_length
        decoded_preds = tokenizer.batch_decode(rc_eval_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(rc_eval_labels, skip_special_tokens=True)

        # calculate F1
        for refs, preds in zip([x.split() for x in decoded_labels], [y.split() for y in decoded_preds]):
            f1_score += compute_f1_from_tokens(refs, preds)

        f1_desc = "Eval F1: {}".format(f1_score/len(decoded_preds))

        # print metrics and update progress bar
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | QR Eval Loss: {qr_eval_metrics['loss']} | {rouge_desc})"
            f" | RC Eval Loss: {rc_eval_metrics['loss']} | {f1_desc})")

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            qr_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], qr_state.params))
            rc_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], rc_state.params))
            print('saving')
            qr_model.save_pretrained(qr_output_dir, params=qr_params)
            rc_model.save_pretrained(rc_output_dir, params=rc_params)
            tokenizer.save_pretrained(output_dir)


