#!/usr/bin/env python
# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license
# found in the LICENSE file in the root directory of this source tree.
import json
import os
import logging
import shutil
from tqdm import tqdm

import torch
import torch.cuda

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from grtr.env_utils import OUTPUT_DIR
from grtr.utils import (get_blank_training_state,
                        save_model)

logger = logging.getLogger(__file__)


def train_epoch(in_model, in_data_loader, in_optimizer, args, training_state, amp=None, save=True):
    in_model.train()
    RESULT_DIR = os.path.join(OUTPUT_DIR, args.model_checkpoint)
    TRAINING_STATE_FILE = os.path.join(RESULT_DIR, 'training_state.json')
    CHECKPOINT_FILE = os.path.join(RESULT_DIR, 'checkpoint.pt')
    for iter, batch in tqdm(enumerate(in_data_loader), total=len(in_data_loader)):
        if iter < training_state['step']:
            continue
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, mc_loss = in_model(*batch)[:2]
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        loss = torch.mean(loss)
        if amp is not None:
            with amp.scale_loss(loss, in_optimizer) as scaled_loss:
                scaled_loss = scaled_loss.float()
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(in_optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(in_model.parameters(), args.max_norm)
        if iter % args.gradient_accumulation_steps == 0:
            in_optimizer.step()
            in_optimizer.zero_grad()
        training_state['step'] += 1
        if args.local_rank in [-1, 0] and args.n_epochs > 0 and save:
            if args.steps_per_checkpoint != 0 and training_state['step'] % args.steps_per_checkpoint == 0:
                save_model(in_model, CHECKPOINT_FILE)
                with open(TRAINING_STATE_FILE, 'w') as training_state_out:
                    json.dump(training_state, training_state_out)


# Evaluation function and evaluator (evaluator output is the input of the metrics)
def evaluate(in_model, in_tokenizer, in_data_loader, args):
    in_model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for batch in tqdm(in_data_loader, total=len(in_data_loader)):
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            logger.debug(in_tokenizer.decode(input_ids[0, -1, :].tolist()))
            model_outputs = in_model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            results = (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
            loss += loss_fn(results[0][0], results[1][0]) / len(in_data_loader)
            batch_acc = ((mc_labels.eq(mc_logits.max(-1)[1])).sum()) / float(mc_labels.shape[0])
            acc += batch_acc / len(in_data_loader)
    return {'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'acc': acc.item() if isinstance(loss, torch.Tensor) else acc}


def train(args,
          model,
          tokenizer,
          optimizer,
          train_loader,
          train_sampler,
          valid_loader,
          valid_sampler,
          save=True,
          amp=None):
    # Training function and trainer
    RESULT_DIR = os.path.join(OUTPUT_DIR, args.model_checkpoint)
    TRAINING_STATE_FILE = os.path.join(RESULT_DIR, 'training_state.json')
    CHECKPOINT_FILE = os.path.join(RESULT_DIR, 'checkpoint.pt')
    BEST_MODEL_FILE = os.path.join(RESULT_DIR, WEIGHTS_NAME)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    training_state = get_blank_training_state()
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE) as training_state_in:
            training_state = json.load(training_state_in)
    if os.path.exists(CHECKPOINT_FILE):
        model.load_state_dict(torch.load(CHECKPOINT_FILE))
    if args.local_rank in [-1, 0] and save:
        torch.save(args, os.path.join(RESULT_DIR, 'model_training_args.bin'))
        getattr(model, 'module', model).config.to_json_file(os.path.join(RESULT_DIR, CONFIG_NAME))
        tokenizer.save_vocabulary(RESULT_DIR)

    for epoch in range(training_state['epoch'], args.n_epochs):
        logger.info('Starting epoch {}'.format(epoch))
        if args.distributed:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
        train_epoch(model, train_loader, optimizer, args, training_state=training_state, amp=amp, save=save)
        eval_dict = evaluate(model, tokenizer, valid_loader, args)
        logger.info(json.dumps(eval_dict, indent=2))

        if args.local_rank in [-1, 0] and args.n_epochs > 0 and save:
            save_model(model, CHECKPOINT_FILE)
        if eval_dict['loss'] < training_state['best_loss']:
            logger.info('New best loss - saving model')
            training_state['best_loss'] = eval_dict['loss']
            training_state['steps_without_improvement'] = 0
            # On the main process: close tensorboard logger and rename the last checkpoint
            # (for easy re-loading with OpenAIGPTModel.from_pretrained method)
            if args.local_rank in [-1, 0] and args.n_epochs > 0:
                if save:
                    shutil.copy(CHECKPOINT_FILE, BEST_MODEL_FILE)
        else:
            training_state['steps_without_improvement'] += 1
        training_state['epoch'] += 1
        training_state['step'] = 0
        with open(TRAINING_STATE_FILE, 'w') as training_state_out:
            json.dump(training_state, training_state_out)
        if training_state['steps_without_improvement'] == args.early_stopping_after:
            logger.info('Stopping after {} epochs'.format(epoch + 1))
            break
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    if os.path.exists(TRAINING_STATE_FILE):
        os.remove(TRAINING_STATE_FILE)
