# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import os
import tarfile
import tempfile
from collections import deque, defaultdict, Counter
from heapq import heappop, heappush, heappushpop
from itertools import chain
from zipfile import ZipFile
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_transformers import cached_path, GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME
from mldc.data.schema import DataSpec
from mldc.preprocessing.stream import stream_dlgs_many
from torch.utils.data import DataLoader, TensorDataset

from grtr.env_utils import OUTPUT_DIR

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

SPECIAL_TOKENS = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'additional_special_tokens': ['<speaker1>', '<speaker2>']
}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)


class MetaLWozDataset(torch.utils.data.Dataset):
    def __init__(self, dialogues, force_min_size=0):

        def valid_dlg(dlg):
            if dlg['domain'] == 'catsstandingup':
                return False
            for turn in dlg['turns']:
                if len(turn) == 0:
                    print("Found invalid dialogue.")
                    return False
            return True

        dataset = [(dlg, dlg['domain'] + "--" + dlg['task_id']) for dlg in dialogues if valid_dlg(dlg)]
        if force_min_size:
            cnt = Counter([d[1] for d in dataset])
            dataset = [(dlg, domain) for dlg, domain in dataset if cnt[domain] >= force_min_size]
        self.items = dataset

    @staticmethod
    def from_dataspec(tokenizer, zipfile_path, dataspec_path, fold, args, dataset_cache, force_min_size=0):
        with open(dataspec_path) as dataspec_in:
            dataspec = DataSpec.load(dataspec_in)
        for fold_name, fold_paths, fold_tasks in zip(['train', 'valid', 'test'],
                                                     dataspec.unpack_paths(), dataspec.unpack_tasks()):
            if fold_name != fold:
                continue
            dialogues = load_metalwoz_dialogues(tokenizer,
                                                zipfile_path,
                                                args,
                                                filenames=fold_paths,
                                                tasks=fold_tasks,
                                                dataset_cache=dataset_cache)
            return MetaLWozDataset(dialogues, force_min_size=force_min_size)

    @staticmethod
    def from_testspec_entry(tokenizer, zipfile_path, testspec_item, args, dataset_cache, filenames=None):

        domain_dialogues = load_metalwoz_dialogues(tokenizer,
                                                   zipfile_path,
                                                   args,
                                                   filenames=filenames,
                                                   dataset_cache=dataset_cache)
        support_dialogues = [dlg for dlg in domain_dialogues if dlg['id'] in testspec_item['support_dlgs']]
        return MetaLWozDataset(support_dialogues)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        return self.items[key]


def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def load_metalwoz_dialogues(tokenizer, zipfile_path, args, filenames=None, tasks=None, dataset_cache=None):
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Tokenize and encode the dataset")

        if filenames is None:
            filenames = [entry for entry in ZipFile(zipfile_path).namelist()
                         if entry.startswith('dialogues/') and entry.endswith('.txt')]
        dataset = []

        tasks = set(tasks if tasks else [])
        for dialogue in stream_dlgs_many(zipfile_path, filenames):
            if tasks and dialogue.task_id not in tasks:
                continue
            turns_tokenized = [tokenizer.encode(' '.join(turn.split()[:args.max_utterance_length]))
                               for turn in dialogue.turns]
            dialogue_json = {'id': dialogue.id,
                             'domain': dialogue.domain,
                             'task_id': dialogue.task_id,
                             'user_id': dialogue.user_id,
                             'bot_id': dialogue.bot_id,
                             'turns': turns_tokenized}
            dataset.append(dialogue_json)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset


def get_metalwoz_dataset(tokenizer, zipfile_path, dataspec_path, dataset_cache=None):
    with open(dataspec_path) as dataspec_in:
        dataspec = DataSpec.load(dataspec_in)
    dataset = {}
    for fold_name, fold_paths, fold_tasks in zip(['train', 'valid', 'test'], dataspec.unpack_paths(), dataspec.unpack_tasks()):
        dataset[fold_name] = load_metalwoz_dialogues(tokenizer,
                                                     zipfile_path,
                                                     None,
                                                     filenames=fold_paths,
                                                     tasks=fold_tasks,
                                                     dataset_cache=dataset_cache)
    return dataset


def populate_candidates_cache(in_dialogues, max_len):
    all_turns = []
    for dialogue in in_dialogues:
        if isinstance(dialogue, tuple):
            dialogue = dialogue[0]  # metadataset instances are (x, domain)
        all_turns += dialogue['turns']
    cache_idx = np.random.choice(list(range(len(all_turns))), size=max_len)
    result = deque([], maxlen=max_len)
    for idx in cache_idx:
        result.append(all_turns[idx])
    return result


def build_input_from_segments(history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['additional_special_tokens'])
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    instance = {}
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence


def slice_dialogue_into_gpt2_input(dialogue, max_history, shared_candidates_cache, num_candidates):
    history = deque([], maxlen=max_history * 2 - 1)
    result = []
    for idx in range(1, len(dialogue), 2):
        history.append(dialogue[idx - 1])
        utterance = dialogue[idx]
        # candiates are: {num_candidates - 1} distractors + gold response
        candidates = []
        while len(candidates) < num_candidates - 1:
            candidate = np.random.choice(shared_candidates_cache)
            if candidate == utterance:
                continue
            candidates.append(candidate)
        candidates.append(utterance)
        shared_candidates_cache.append(utterance)
        result.append((list(history), candidates))
        history.append(utterance)
    return result


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def pad_batch(batch, num_candidates, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """

    max_l = max(len(cand) for item in batch for cand in item['input_ids'])
    padded = dict()
    for name in PADDED_INPUTS:
        padded[name] = np.full((len(batch), num_candidates, max_l), padding if name != 'lm_labels' else -1)
        for batch_idx, item in enumerate(batch):
            assert len(item[name]) == num_candidates
            for cand_idx, cand in enumerate(item[name]):
                padded[name][batch_idx, cand_idx, :len(cand)] = cand

    tensorized_batch = []
    for input_name in MODEL_INPUTS:
        if input_name in padded:
            tensor = torch.tensor(padded[input_name])
        else:
            tensor = torch.tensor([x[input_name] for x in batch])
        tensorized_batch.append(tensor)
    return tensorized_batch


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_loader_for_dataset(dataset,
                           tokenizer,
                           max_history,
                           num_candidates,
                           batch_size,
                           dataloader_num_workers=0,
                           distributed=False,
                           max_samples=0):
    candidates_cache = populate_candidates_cache(dataset, max_len=1000)

    logger.info("Build inputs and labels")
    featurized_dataset = defaultdict(list)
    for dialog_json in dataset:
        dialogue = dialog_json['turns']
        if len(dialogue) < 2:
            continue
        for history, candidates in slice_dialogue_into_gpt2_input(dialogue,
                                                                  max_history,
                                                                  candidates_cache,
                                                                  num_candidates):
            for j, candidate in enumerate(candidates):
                lm_labels = bool(j == num_candidates - 1)
                instance, _ = build_input_from_segments(history, candidate, tokenizer, lm_labels)
                for input_name, input_array in instance.items():
                    featurized_dataset[input_name].append(input_array)
            featurized_dataset["mc_labels"].append(num_candidates - 1)
            featurized_dataset["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    tensorized_dataset = []
    dataset_padded = pad_dataset(featurized_dataset, padding=tokenizer.pad_token_id)
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(dataset_padded[input_name])
        if input_name != "mc_labels":
            tensor = tensor.view((-1, featurized_dataset["n_candidates"]) + tensor.shape[1:])
        tensorized_dataset.append(tensor)

    if 0 < max_samples:
        # shuffling and trimming to max_samples
        shuffle_index = list(range(len(featurized_dataset['mc_labels'])))
        np.random.shuffle(shuffle_index)
        shuffle_index = shuffle_index[:max_samples]
        tensorized_dataset_trimmed = []
        for tensor_i in tensorized_dataset:
            tensorized_dataset_trimmed.append(tensor_i[shuffle_index, ...])
        tensorized_dataset = tensorized_dataset_trimmed

    logger.info("Build dataloader")
    tensor_dataset = TensorDataset(*tensorized_dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(tensor_dataset) if distributed else None
    loader = DataLoader(tensor_dataset,
                        sampler=sampler,
                        batch_size=batch_size,
                        num_workers=dataloader_num_workers,
                        shuffle=(not distributed),
                        drop_last=True)
    return loader, sampler


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['additional_special_tokens']) + [
        tokenizer.eos_token_id, tokenizer.bos_token_id + tokenizer.pad_token_id
    ]
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)[0]

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            for _ in range(10):
                prev = torch.multinomial(probs, num_samples=1)
                if prev.item() not in special_tokens_ids:
                    break

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def generate_and_rank(support_contexts,
                      support_responses,
                      target_context,
                      tokenizer,
                      model,
                      encoder,
                      args,
                      ret_winner=False,
                      encodings_cache={}):
    generated = sample_sequence(target_context, tokenizer, model, args)
    winner = 'generated'
    if not len(support_contexts):
        return (generated, winner) if ret_winner else generated
    target_context_emb = embed_dialogue(target_context,
                                        None,
                                        tokenizer,
                                        encoder,
                                        args,
                                        encodings_cache=encodings_cache)

    num_ret_candidates = args.num_candidates - 1
    candidates_heap = []
    for idx, support_context in enumerate(support_contexts):
        support_context_emb = embed_dialogue(support_context,
                                             None,
                                             tokenizer,
                                             encoder,
                                             args,
                                             encodings_cache=encodings_cache)
        distance = target_context_emb.dist(support_context_emb)
        if len(candidates_heap) < num_ret_candidates:
            heappush(candidates_heap, (-distance, idx))
        else:
            heappushpop(candidates_heap, (-distance, idx))
    if len(candidates_heap) == 0:
        return (generated, winner) if ret_winner else generated

    ret_instances = []
    candidates = []
    while len(ret_instances) != num_ret_candidates:
        if len(candidates_heap) == 0:
            ret_instances.append(copy.deepcopy(ret_instances[-1]))
            candidates.append(copy.deepcopy(candidates[-1]))
            continue
        _, idx = heappop(candidates_heap)
        ret_instance, _ = build_input_from_segments(target_context,
                                                    support_responses[idx],
                                                    tokenizer,
                                                    with_eos=True)
        ret_instances.append(ret_instance)
        candidates.append(support_responses[idx])

    gen_instance, _ = build_input_from_segments(target_context, generated, tokenizer, with_eos=True)
    candidates.append(generated)

    data_point = defaultdict(list)
    for instance in ret_instances + [gen_instance]:
        for key, value in instance.items():
            data_point[key].append(value)
    data_point["mc_labels"].append(args.num_candidates - 1)
    data_point["num_candidates"] = args.num_candidates

    tensorized_data_point = []
    dataset_padded = pad_dataset(data_point, padding=tokenizer.pad_token_id)
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(dataset_padded[input_name])
        if input_name != "mc_labels":
            tensor = tensor.view((-1, data_point["num_candidates"]) + tensor.shape[1:])
        if args.device == 'cuda':
            tensor = tensor.cuda()
        tensorized_data_point.append(tensor)

    model_output = model(*tensorized_data_point)
    mc_labels = model_output[3]
    arg_max = mc_labels.max(-1)[1]
    if arg_max != args.num_candidates - 1:
        winner = 'retrieved'
    logger.info(f"{winner} response won")

    return (candidates[arg_max], winner) if ret_winner else candidates[arg_max]


def embed_dialogue(context, response, tokenizer, encoder, args, encodings_cache={}):
    if response is None:
        response = []

    dialogue_id = hashlib.sha256(str(context + response).encode('utf-8')).digest()
    if dialogue_id in encodings_cache:
        logging.info('Encodings cache hit')
        logging.info(str(context + response) + ' --> ' + str(dialogue_id))
        return encodings_cache[dialogue_id]

    instance, sequence = build_input_from_segments(context, response, tokenizer, with_eos=True)
    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
    cls_index = torch.tensor(instance["mc_token_ids"], device=args.device).unsqueeze(0)

    emb = encoder(input_ids, token_type_ids=token_type_ids)[0]
    cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
    cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (emb.size(-1),))
    # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
    emb = emb.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)

    encodings_cache[dialogue_id] = emb
    return emb


def get_blank_training_state():
    return {'epoch': 0, 'step': 0, 'best_loss': np.inf, 'steps_without_improvement': 0}


def load_tokenizer_and_model(in_args):
    tokenizer_class = GPT2Tokenizer
    model_class = GPT2DoubleHeadsModel
    checkpoint_full_path = os.path.join(OUTPUT_DIR, in_args.model_checkpoint)
    weights_full_path = os.path.join(checkpoint_full_path, WEIGHTS_NAME)
    checkpoint_to_load = checkpoint_full_path if os.path.exists(weights_full_path) else in_args.model_name

    tokenizer = tokenizer_class.from_pretrained(checkpoint_to_load)
    model = model_class.from_pretrained(checkpoint_to_load)
    return tokenizer, model


def save_model(in_model, in_dst_file):
    torch.save(in_model.state_dict(), in_dst_file)


def load_model(in_model, in_src_file):
    in_model.load_state_dict(torch.load(in_src_file))
