""" Utility for fine-tuning LMs on the translation and co-reference resolution task """

from __future__ import absolute_import, division, print_function

import os
import re
import glob
import json
import torch
import random
import shutil
import logging
import sacrebleu

import numpy as np

from io import open
from sacrerouge.metrics import Rouge
from torch.nn import CrossEntropyLoss
from comet.models import download_model

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask


class MTExample(object):
    """A single training/test example for the translation task."""

    def __init__(self, src, tgt):
        self.type = 'mt'
        self.src = src
        self.tgt = tgt


class WinoGrandeQAExample(object):
    """A single training/test example for the WinoGrande QA task."""

    def __init__(self, guid, context, question, option1, option2, answer):
        self.type = 'qa'
        self.guid = guid
        self.context = context
        self.question = question
        self.option1 = option1
        self.option2 = option2
        self.answer = answer


class WinoGrandeMLMExample(object):
    """A single training/test example for the WinoGrande MLM task."""

    def __init__(self, guid, context, option1, option2, answer):
        self.type = 'mlm'
        self.guid = guid
        self.context = context
        self.option1 = option1
        self.option2 = option2
        self.answer = answer


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, args):
        self.query_lang = args.query_lang
        self.answer_lang = args.answer_lang

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, src_file, tgt_file):
        """Reads a plain-text file."""
        records = []
        logging.info('Reading in parallel data ...')
        with open(src_file, "r", encoding="utf-8") as src_f:
            src_lines = src_f.readlines()
        with open(tgt_file, "r", encoding="utf-8") as tgt_f:
            tgt_lines = tgt_f.readlines()
        for line_id in range(len(src_lines)):
            records.append({'src': src_lines[line_id].strip(), 'tgt': tgt_lines[line_id].strip()})
        return records

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a .jsonl file."""
        records = []
        logging.info('Reading in data ...')
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return records


class MTProcessor(DataProcessor):
    """ Converts a parallel corpus for use with cross-lingual LMs. """

    def get_train_examples(self, data_dir):
        assert self.query_lang is not None and self.answer_lang is not None, \
            'Source and target languages are not specified!'
        return self._create_examples(
            self._read_txt('{:s}/train.{:s}}'.format(data_dir, self.query_lang),
                           '{:s}/train.{:s}}'.format(data_dir, self.answer_lang)))

    def get_dev_examples(self, data_dir):
        assert self.query_lang is not None and self.answer_lang is not None, \
            'Source and target languages are not specified!'
        return self._create_examples(
            self._read_txt('{:s}/dev.{:s}}'.format(data_dir, self.query_lang),
                           '{:s}/dev.{:s}}'.format(data_dir, self.answer_lang)))

    def get_test_examples(self, data_dir):
        assert self.query_lang is not None and self.answer_lang is not None, \
            'Source and target languages are not specified!'
        return self._create_examples(
            self._read_txt('{:s}/test.{:s}}'.format(data_dir, self.query_lang),
                           '{:s}/test.{:s}}'.format(data_dir, self.answer_lang)))

    def get_labels(self):
        return None

    def create_examples(self, records):
        return self._create_examples(records)

    @staticmethod
    def _create_examples(records):
        # Convert corpus contents to examples
        examples = list()
        for i, record in enumerate(records):
            src_line = record.get('src', None)
            tgt_line = record.get('tgt', None)
            examples.append(MTExample(src=src_line, tgt=tgt_line))

        return examples


class WinoGrandeQAProcessor(DataProcessor):
    """ Prepares WinoGrande QA data for use with LMs. """

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'train.jsonl')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'dev.jsonl')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'test.jsonl')))

    def get_labels(self):
        return None

    def create_examples(self, records):
        return self._create_examples(records)

    def _create_examples(self, records):
        # Convert corpus contents to examples
        examples = list()
        for i, record in enumerate(records):
            qid = record.get('qID', None)
            context = record.get('sentence', None)
            question = record.get('question', None)
            option1 = record.get('option1_{:s}'.format(self.answer_lang))
            option2 = record.get('option2_{:s}'.format(self.answer_lang))
            answer = record.get('answer')
            examples.append(WinoGrandeQAExample(guid=qid, context=context, question=question,
                                                option1=option1, option2=option2, answer=answer))

        return examples


class WinoGrandeMLMProcessor(DataProcessor):
    """ Prepares WinoGrande MLM data for use with LMs. """

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'train.jsonl')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'dev.jsonl')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'test.jsonl')))

    def get_labels(self):
        return None

    def create_examples(self, records):
        return self._create_examples(records)

    def _create_examples(self, records):
        # Convert corpus contents to examples
        examples = list()
        for i, record in enumerate(records):
            qid = record.get('qID', None)
            if 'tgt_context' not in record.keys():
                context = record.get('context_{:s}'.format(self.query_lang), None)
                option1 = record.get('option1_{:s}'.format(self.answer_lang))
                option2 = record.get('option2_{:s}'.format(self.answer_lang))
            else:
                context = record.get('tgt_context', None)
                option1 = record.get('option1')
                option2 = record.get('option2')
            answer = record.get('answer')
            examples.append(WinoGrandeMLMExample(guid=qid, context=context,
                                                 option1=option1, option2=option2, answer=answer))

        return examples


def convert_examples_to_features(examples,
                                 max_seq_length,
                                 tokenizer,
                                 model_name,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 fit_to_max_corpus_len=True):
    """ Loads a data file into a list of input batches """

    # Initialize containers
    src_cache = list()
    tgt_cache = list()

    # Tokenize
    lines = list()
    for ex_index, ex in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))

        # Construct and tokenize sample contents
        if ex.type == 'mt':
            tokens_src = tokenizer.tokenize(ex.src)
            with tokenizer.as_target_tokenizer():
                tokens_tgt = tokenizer.tokenize(ex.tgt)
            example_tokens = (tokens_src, tokens_tgt)
        elif ex.type == 'qa':
            context_tokens = tokenizer.tokenize(ex.sentence)
            question_tokens = tokenizer.tokenize(ex.question)
            with tokenizer.as_target_tokenizer():
                tokens_tgt = tokenizer.tokenize(ex.option1) if ex.answer == 1 else tokenizer.tokenize(ex.option2)
            example_tokens = (context_tokens, question_tokens, tokens_tgt)
        else:
            if model_name not in ['t5', 'mt5']:
                # Replace gap identifier with model-specific mask token
                src = ex.context.replace('_', tokenizer.mask_token)
                tokens_src = tokenizer.tokenize(src)
                tgt = ex.context.replace('_', ex.option1) if ex.answer == 1 else ex.context.replace('_', ex.option2)
                with tokenizer.as_target_tokenizer():
                    tokens_tgt = tokenizer.tokenize(tgt)
            else:
                src = ex.context.replace('_', '<extra_id_0>')
                tokens_src = tokenizer.tokenize(src)
                tgt = '<extra_id_0> {:s} <extra_id_1>'.format(ex.option1) if ex.answer == 1 else \
                    '<extra_id_0> {:s} <extra_id_1>'.format(ex.option2)
                with tokenizer.as_target_tokenizer():
                    tokens_tgt = tokenizer.tokenize(tgt)
            example_tokens = (tokens_src, tokens_tgt)
        lines.append(example_tokens)

    # Optionally truncate inputs to max_seq_len
    for example_tokens in lines:
        special_tokens_count = 2 if model_name not in ['t5', 'mt5'] else 1
        if len(example_tokens) > 2:
            special_tokens_count += 1 if model_name not in ['roberta_en', 'roberta_fr', 'xlm-r'] else 2
        # Truncate inputs, if needed
        if not fit_to_max_corpus_len:
            _truncate_seq_pair(example_tokens, max_seq_length - special_tokens_count - 1)

        if len(example_tokens) > 2:
            separator_list = \
                [tokenizer.sep_token] if model_name not in ['roberta_en', 'roberta_fr', 'xlm-r'] \
                else [tokenizer.sep_token] * 2
            src_tokens = example_tokens[0] + separator_list + example_tokens[1]
            src_segments = [0] * (len(example_tokens[0]) + 1) + [1] * (len(src_tokens) - (len(example_tokens[0]) + 1))
            tgt_tokens = example_tokens[-1]
            # Account for special tokens
            if model_name not in ['t5', 'mt5']:
                src_segments = [0] + src_segments + [1]
            else:
                src_segments = src_segments + [1]
        else:
            src_tokens, tgt_tokens = example_tokens[-1]
            if model_name not in ['t5', 'mt5']:
                src_segments = [1] * (len(src_tokens) + 2)
            else:
                src_segments = [1] * (len(src_tokens) + 1)

        # Encode
        encoded_src = tokenizer.encode(src_tokens)  # returns a list of indices, including special tokens
        assert len(encoded_src) == len(src_segments), \
            'Length mismatch: length of encoded SRC is {:d}, length of SRC segment IDs is {:d}'.format(
                len(encoded_src), len(src_segments))
        src_cache.append((src_tokens, encoded_src, src_segments))
        with tokenizer.as_target_tokenizer():
            encoded_tgt = tokenizer.encode(tgt_tokens)
        tgt_cache.append((tgt_tokens, encoded_tgt))

    # Determine maximum input tokens length
    src_lengths = [len(tpl[1]) for tpl in src_cache]
    max_src_length = max(src_lengths)
    tgt_lengths = [len(tpl[1]) for tpl in tgt_cache]
    max_tgt_length = max(tgt_lengths)

    # Make masks and pad inputs / labels
    features = list()
    for iid, inputs in enumerate(src_cache):
        src_tokens, src_ids, src_segments = inputs
        tgt_tokens, tgt_ids = tgt_cache[iid]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        src_mask = [1 if mask_padding_with_zero else 0] * len(src_ids)
        tgt_mask = [1 if mask_padding_with_zero else 0] * len(tgt_ids)

        # Zero-pad up to the sequence length
        src_padding_length = max_src_length - len(src_ids)
        if pad_on_left:
            src_ids = ([pad_token] * src_padding_length) + src_ids
            src_mask = ([0 if mask_padding_with_zero else 1] * src_padding_length) + src_mask
            src_segments = ([pad_token_segment_id] * src_padding_length) + src_segments
        else:
            src_ids = src_ids + ([pad_token] * src_padding_length)
            src_mask = src_mask + ([0 if mask_padding_with_zero else 1] * src_padding_length)
            src_segments = src_segments + ([pad_token_segment_id] * src_padding_length)

        # Mask out padding positions, so that they don't contribute to the loss calculation
        tgt_padding_length = max_tgt_length - len(tgt_ids)
        if pad_on_left:
            tgt_ids = ([-100] * tgt_padding_length) + tgt_ids
            tgt_mask = ([0 if mask_padding_with_zero else 1] * tgt_padding_length) + tgt_mask
        else:
            tgt_ids = tgt_ids + ([-100] * tgt_padding_length)
            tgt_mask = tgt_mask + ([0 if mask_padding_with_zero else 1] * tgt_padding_length)

        try:
            assert len(src_ids) == max_src_length
            assert len(src_mask) == max_src_length
            assert len(src_segments) == max_src_length
        except AssertionError:
            logging.info(src_ids, len(src_ids))
            logging.info(src_mask, len(src_mask))
            logging.info(src_segments, len(src_segments))
            raise AssertionError

        if iid < 5:
            logger.info("*** Example ***")
            logger.info("src_tokens: %s" % " ".join(src_tokens))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("src_mask: %s" % " ".join([str(x) for x in src_mask]))
            logger.info("src_segments: %s" % " ".join([str(x) for x in src_segments]))
            logger.info('-' * 20)
            logger.info("tgt_tokens: %s" % " ".join(tgt_tokens))
            logger.info("tgt_ids: %s" % " ".join([str(x) for x in tgt_ids]))
            logger.info("tgt_mask: %s" % " ".join([str(x) for x in tgt_mask]))

        features.append(
            InputFeatures(input_ids=src_ids,
                          input_mask=src_mask,
                          segment_ids=src_segments,
                          label_ids=tgt_ids,
                          label_mask=tgt_mask))

    # Report some basic statistics
    logger.info('=' * 20)
    logger.info('Dataset statistics (before truncation / padding):')
    logger.info('Mean model input length: {:.2f}'.format(np.mean(src_lengths)))
    logger.info('Model input length std.: {:.2f}'.format(np.std(src_lengths)))
    logger.info('Min model input length: {:.2f}'.format(min(src_lengths)))
    logger.info('Max model input length: {:.2f}'.format(max(src_lengths)))
    logger.info('-' * 20)
    logger.info('Mean model target length: {:.2f}'.format(np.mean(tgt_lengths)))
    logger.info('Model target length std.: {:.2f}'.format(np.std(tgt_lengths)))
    logger.info('Min model target length: {:.2f}'.format(min(tgt_lengths)))
    logger.info('Max model target length: {:.2f}'.format(max(tgt_lengths)))
    logger.info('=' * 20)

    return features


def _truncate_seq_pair(all_segments, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # Don't truncate the target tokens
    final_segment = [all_segments[-1]]
    all_segments = all_segments[:-1]

    while True:
        total_length = sum([len(seg) for seg in all_segments])
        if total_length <= max_length:
            break
        # Shorten the longest segment
        longest_seg = all_segments[max(enumerate(all_segments), key=lambda x: len(x[1]))[0]]
        longest_seg.pop()

    all_segments += final_segment


def set_seed(args):
    """ Sets the seed to support reproducibility. """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    """ Keep a maximum of args.save_total_limit checkpoints (adopted from the SC101 scripts). """
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = list()
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info('Deleting older checkpoint [{}] due to args.save_total_limit'.format(checkpoint))
        shutil.rmtree(checkpoint)


def get_token_loss(lm_logits, target_ids, target_mask):
    """ Compute token-level loss per batch during evaluation (adopted from the SC101 scripts).
     NOTE: There is some redundant computation going on here which should be fixed at some point. """
    # Declare loss function
    loss_fct = CrossEntropyLoss(reduction='none')

    # Obtain logits to compute token-level loss / perplexity
    batch_size, max_length, vocab_size = lm_logits.shape

    # Compute loss for each instance and each token
    lm_logits = lm_logits.view(-1, vocab_size)
    lm_labels = target_ids[:, 1:].clone().contiguous().view(-1)
    token_loss = loss_fct(lm_logits, lm_labels).view(batch_size, max_length)

    # Only consider non padded tokens
    target_mask = target_mask[..., :-1].contiguous()
    masked_token_loss = torch.mul(target_mask, token_loss)  # [batch_size, max_length]

    return masked_token_loss


## METRICS
def compute_bleu(preds, targets):
    """ Computes corpus-level BLEU for the generated sequences. """
    targets = [targets]
    bleu = sacrebleu.corpus_bleu(preds, targets)
    return bleu.score


def compute_rouge(preds, targets):
    """ Computes ROUGE-L for the generated sequences. """
    rouge = Rouge(compute_rouge_l=True)
    rouge_out = rouge.evaluate(preds, [[tgt] for tgt in targets])
    return rouge_out[0]['rouge-l']['f1']


def compute_comet(sources, preds, targets):
    """ Computes the COMET score. """
    model = download_model('wmt-large-da-estimator-1719')
    data = {'src': sources, 'mt': preds, 'ref': targets}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    out = model.predict(data, cuda=True, show_progress=True)
    return np.mean(out)


def compute_gen_metrics(sources, preds, targets):
    """ Aggregates generation metrics. """
    assert len(preds) == len(targets)
    return {'BLEU-4': str(compute_bleu(preds, targets)),
            'ROUGE-L': str(compute_rouge(preds, targets)),
            'COMET': str(compute_comet(sources, preds, targets))}

