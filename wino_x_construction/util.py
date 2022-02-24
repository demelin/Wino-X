""" Various helper scripts. """
import os
import re
import json
import glob
import torch
import shutil
import string
import random
import logging
import sacrebleu

import numpy as np

from torch.nn import CrossEntropyLoss

from spacy_tag_map import TAG_MAP

logger = logging.getLogger(__name__)


def read_jsonl(input_file):
    """ Helper for reading .jsonl files """
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def list_symmetric_difference(list1, list2):
    """ Computes the symmetric difference between two lists. """
    list1_copy = list1[:]
    list2_copy = list2[:]
    for i in list1:
        if i in list2:
            list2.remove(i)
    for j in list2_copy:
        if j in list1_copy:
            list1_copy.remove(j)
    return list2 + list1_copy


def _map_spacy_to_whitespace_tokens(spacy_tokens, ws_tokens):
    """ Constructs a map between tokens obtained from spaCy tokenizer and from whitespace tokenization. """
    # Map whitespace tokens to spacy tokens
    ws_to_spacy_map = dict()
    spacy_to_ws_map = dict()
    ws_loc = 0
    ws_tok = ws_tokens[ws_loc]
    for spacy_loc, spacy_tok in enumerate(spacy_tokens):
        while True:
            if spacy_tok == ws_tok or spacy_tok in ws_tok:
                # Terminate
                if ws_loc >= len(ws_tokens):
                    break

                # Extend maps
                if not ws_to_spacy_map.get(ws_loc, None):
                    ws_to_spacy_map[ws_loc] = list()
                ws_to_spacy_map[ws_loc].append(spacy_loc)
                if not spacy_to_ws_map.get(spacy_loc, None):
                    spacy_to_ws_map[spacy_loc] = list()
                spacy_to_ws_map[spacy_loc].append(ws_loc)

                # Move pointer
                if spacy_tok == ws_tok:
                    ws_loc += 1
                    if ws_loc < len(ws_tokens):
                        ws_tok = ws_tokens[ws_loc]
                else:
                    ws_tok = ws_tok[len(spacy_tok):]
                break
            else:
                ws_loc += 1

    # Assert full coverage of whitespace and SpaCy token sequences by the mapping
    ws_covered = sorted(list(ws_to_spacy_map.keys()))
    spacy_covered = sorted(list(set(list([val for val_list in ws_to_spacy_map.values() for val in val_list]))))
    assert ws_covered == [n for n in range(len(ws_tokens))], \
        'WS-SpaCy mapping does not cover all whitespace tokens: {}; number of tokens: {}' \
            .format(ws_covered, len(ws_tokens))
    assert spacy_covered == [n for n in range(len(spacy_tokens))], \
        'WS-SpaCy mapping does not cover all SpaCy tokens: {}; number of tokens: {}' \
            .format(spacy_covered, len(spacy_tokens))
    return ws_to_spacy_map, spacy_to_ws_map


def _connect_sentences(sentence_parts, joined_predictions, scorer, connectors, forced_connector=None):
    """ Helper function that connects several sentences into a single one using various connectors and selecting the
    most fluent result. """
    # Join using connectors
    if forced_connector is not None:
        connectors = [forced_connector]  # ensures that true and false samples use the same connector
    merged_sentences = [c.join(sentence_parts) for c in connectors]
    # Pick best connector
    sentence_scores = scorer.score_sentences(merged_sentences)
    sorted_sentences = \
        sorted(list(zip(merged_sentences, sentence_scores, connectors)), reverse=True, key=lambda x: x[1])
    if joined_predictions[0] is None:
        joined_predictions[0] = sorted_sentences
    else:
        first_predictions = joined_predictions[0]
        second_predictions = sorted_sentences
        first_joined_sentence, first_score, first_connector = first_predictions[0]
        second_joined_sentence, second_score, second_connector = second_predictions[0]
        if first_connector != second_connector:
            best_connector = first_connector if first_score >= second_score else second_connector
            for tpl_id, tpl in enumerate(first_predictions):
                if tpl[2] == best_connector:
                    best_tpl = first_predictions.pop(tpl_id)
                    first_predictions = [best_tpl] + first_predictions
                    break
            for tpl_id, tpl in enumerate(second_predictions):
                if tpl[2] == best_connector:
                    best_tpl = second_predictions.pop(tpl_id)
                    second_predictions = [best_tpl] + second_predictions
                    break
        joined_predictions = [first_predictions, second_predictions]
    return joined_predictions


def parse_sample_pair(pair, nlp, scorer, connectors, exclude_multi_sentence_samples):
    """ Parses WinoGrande sentences (two per sample) and annotates them. """
    sample_features = list()

    true_referents = [None, None]
    true_referent_lens = [None, None]
    true_filled_sentences = [None, None]
    true_init_gap_ids = [None, None]
    true_joined_predictions = [None, None]

    false_referents = [None, None]
    false_referent_lens = [None, None]
    false_filled_sentences = [None, None]
    false_init_gap_ids = [None, None]
    false_joined_predictions = [None, None]

    is_multi_sentence = [False, False]

    # Handle multi-sentence inputs
    for filled_sentences, gap_ids, referents, referent_lens, joined_predictions, is_true in \
            [(true_filled_sentences, true_init_gap_ids, true_referents, true_referent_lens,
              true_joined_predictions, True),
             (false_filled_sentences, false_init_gap_ids, false_referents, false_referent_lens,
              false_joined_predictions, False)]:

        for s_id in [0, 1]:

            sample = pair[s_id]
            sample['sentence'] = sample['sentence'].replace('\'s', ' \'s').replace('  ', ' ')
            # Fill gap with true referent
            init_ws_tokens = sample['sentence'].strip().split()
            if is_true:
                referents[s_id] = sample['option1'] if sample['answer'] == '1' else sample['option2']
            else:
                referents[s_id] = sample['option2'] if sample['answer'] == '1' else sample['option1']
            referent_lens[s_id] = len(referents[s_id].split())

            gap_ids[s_id] = init_ws_tokens.index('_')
            ws_tokens = [t for t in init_ws_tokens]
            ws_tokens[gap_ids[s_id]] = referents[s_id]
            filled_sentence = ' '.join(ws_tokens)

            # Combine multi-sentence sentences if necessary (limit to two sentences at most)
            sentence_parts = [p for p in filled_sentence.split('.') if len(p.strip()) > 0]

            if len(sentence_parts) == 2 and exclude_multi_sentence_samples:
                return None

            if len(sentence_parts) == 2 and len(connectors) > 0 and scorer is not None:

                true_connector = None if is_true else true_joined_predictions[0][0][2]
                is_multi_sentence[s_id] = True
                sentence_parts_sub_tokens = sentence_parts[1].split()
                sentence_parts_sub_tokens[0] = \
                    sentence_parts_sub_tokens[0].lower() if referents[s_id][0] == referents[s_id][0].lower() \
                        else sentence_parts_sub_tokens[0]
                sentence_parts[1] = ' '.join(sentence_parts_sub_tokens)
                if sentence_parts[1].endswith('_'):
                    sentence_parts[1] = sentence_parts[1] + ' .'
                else:
                    sentence_parts[1] = sentence_parts[1] + '.'

                joined_predictions = _connect_sentences(sentence_parts, joined_predictions, scorer, connectors,
                                                        forced_connector=true_connector)
                # Note: Hack used to update the respective joined_prediction values (do not remove)
                if is_true:
                    true_joined_predictions = joined_predictions
                else:
                    false_joined_predictions = joined_predictions

            else:
                filled_sentences[s_id] = filled_sentence

    if is_multi_sentence[0] != is_multi_sentence[1] and len(connectors) > 0:
        return None

    # Get features
    for filled_sentences, gap_ids, referents, referent_lens, joined_predictions, is_true in \
            [(true_filled_sentences, true_init_gap_ids, true_referents, true_referent_lens,
              true_joined_predictions, True),
             (false_filled_sentences, false_init_gap_ids, false_referents, false_referent_lens,
              false_joined_predictions, False)]:

        for s_id in [0, 1]:

            sample = pair[s_id]
            referent = referents[s_id]
            referent_len = referent_lens[s_id]
            init_gap_id = gap_ids[s_id]

            # Re-introduce gap
            if filled_sentences[s_id] is None:
                filled_sentence = joined_predictions[s_id][0][0]
                num_extra_tokens = joined_predictions[s_id][0][2].count(' ') - 1
                init_gap_id += num_extra_tokens
            else:
                filled_sentence = filled_sentences[s_id]
            filled_sentence_tokens = filled_sentence.split()
            for _ in range(referent_len - 1):
                del filled_sentence_tokens[init_gap_id]
            filled_sentence_tokens[init_gap_id] = '_'
            init_ws_tokens = filled_sentence_tokens

            # Account for multi-word referents
            if referent_len > 1:
                ws_gap_ids = [init_gap_id + i for i in range(referent_len)]
            else:
                ws_gap_ids = [init_gap_id]
            ws_tokens = filled_sentence.split()  # to tokenize multi-word referents

            # Parse filled sentence with spaCy
            doc = nlp(filled_sentence)
            spacy_tokens = [tok.text for tok in doc]
            ptb_tags = [tok.tag_ for tok in doc]
            pos_tags = [tok.pos_ for tok in doc]
            dep_tags = [tok.dep_ for tok in doc]
            morph_features = [TAG_MAP[tag] for tag in ptb_tags]
            ws_spacy_map, spacy_ws_map = _map_spacy_to_whitespace_tokens(spacy_tokens, ws_tokens)
            spacy_gap_ids = list()
            for i in ws_gap_ids:
                spacy_gap_ids += ws_spacy_map[i]

            # Remove punctuation
            spacy_gap_ids = [i for i in spacy_gap_ids if spacy_tokens[i].strip() not in string.punctuation]
            # Identify noun positions, including that of denoted referents
            spacy_noun_loc = [i for i in range(len(pos_tags)) if pos_tags[i] in ['NOUN', 'PROPN']]
            ws_noun_loc = list()
            for loc in spacy_noun_loc:
                ws_noun_loc += spacy_ws_map[loc]
            ws_noun_tokens = [ws_tokens[i] for i in ws_noun_loc]
            # If options are not present in the sentence, ignore sentence
            ws_tokens_lower = \
                [t.lower().translate(str.maketrans('', '', string.punctuation.replace('-', ''))) for t in ws_tokens]
            ws_tokens_with_gap = sample['sentence'].strip().split()
            ws_tokens_lower_with_gap = [t.lower().translate(str.maketrans(
                '', '', string.punctuation.replace('-', ''))) for t in ws_tokens_with_gap]
            ws_option1_loc = list()
            ws_option2_loc = list()
            option1_tokens = [t for t in sample['option1'].lower().split() if len(t) > 0]
            option2_tokens = [t for t in sample['option2'].lower().split() if len(t) > 0]
            # Locate referent mentions
            for opt1_tok in option1_tokens:
                for t_id, t in enumerate(ws_tokens_lower_with_gap):
                    if len(' '.join(list_symmetric_difference(list(opt1_tok), list(t))).strip().translate(
                            str.maketrans('', '', string.punctuation))) == 0:
                        ws_option1_loc.append(t_id)
            for opt2_tok in option2_tokens:
                for t_id, t in enumerate(ws_tokens_lower_with_gap):
                    if len(' '.join(list_symmetric_difference(list(opt2_tok), list(t))).strip().translate(
                            str.maketrans('', '', string.punctuation))) == 0:
                        ws_option2_loc.append(t_id)
            ws_option1_loc = sorted(ws_option1_loc)
            ws_option2_loc = sorted(ws_option2_loc)
            ws_option1_loc = [ws_option1_loc[0]] if len(ws_option1_loc) == 2 and \
                                                    abs(ws_option1_loc[0] - ws_option1_loc[1]) != 1 else ws_option1_loc
            ws_option2_loc = [ws_option2_loc[0]] if len(ws_option2_loc) == 2 and \
                                                    abs(ws_option2_loc[0] - ws_option2_loc[1]) != 1 else ws_option2_loc

            # Remove wrong locations
            if len(option1_tokens) < len(ws_option1_loc):
                kept_loc = list()
                for l_i, l in enumerate(ws_option1_loc):
                    if l_i == 0:
                        if ws_option1_loc[l_i + 1] != l + 1:
                            continue
                    elif l_i == len(ws_option1_loc) - 1:
                        if ws_option1_loc[l_i - 1] != l - 1:
                            continue
                    else:
                        if ws_option1_loc[l_i + 1] != l + 1 and ws_option1_loc[l_i - 1] != l - 1:
                            continue
                    kept_loc.append(l)
                ws_option1_loc = kept_loc

            if len(option2_tokens) < len(ws_option2_loc):
                kept_loc = list()
                for l_i, l in enumerate(ws_option2_loc):
                    if l_i == 0:
                        if ws_option2_loc[l_i + 1] != l + 1:
                            continue
                    elif l_i == len(ws_option2_loc) - 1:
                        if ws_option2_loc[l_i - 1] != l - 1:
                            continue
                    else:
                        if ws_option2_loc[l_i + 1] != l + 1 and ws_option2_loc[l_i - 1] != l - 1:
                            continue
                    kept_loc.append(l)
                ws_option2_loc = kept_loc

            if len(option1_tokens) != len(ws_option1_loc):
                subsequences = list()
                temp = list()
                for l_id, l in enumerate(ws_option1_loc):
                    if len(temp) == 0:
                        temp.append(l)
                    elif ws_option1_loc[l_id - 1] == l - 1:
                        temp.append(l)
                    else:
                        subsequences.append(temp)
                        temp = [l]
                for ss in subsequences:
                    if len(ss) == len(option1_tokens):
                        ws_option1_loc = ss

            if len(option2_tokens) != len(ws_option2_loc):
                subsequences = list()
                temp = list()
                for l_id, l in enumerate(ws_option2_loc):
                    if len(temp) == 0:
                        temp.append(l)
                    elif ws_option2_loc[l_id - 1] == l - 1:
                        temp.append(l)
                    else:
                        subsequences.append(temp)
                        temp = [l]
                for ss in subsequences:
                    if len(ss) == len(option2_tokens):
                        ws_option2_loc = ss

            # Fill gap with ambiguous pronoun
            is_singular = morph_features[spacy_gap_ids[-1]].get('Number_sing', False)
            deleted_det = False
            if init_ws_tokens[init_gap_id - 1].lower() in ['the', 'a', 'an']:
                del init_ws_tokens[init_gap_id - 1]
                init_gap_id -= 1
                deleted_det = True
            if ws_tokens[init_gap_id - 1].endswith('.'):
                init_ws_tokens[init_gap_id] = 'It' if is_singular else 'They'
            else:
                init_ws_tokens[init_gap_id] = 'it' if is_singular else 'they'

            ambiguous_sentence = ' '.join(init_ws_tokens)
            if init_gap_id == len(init_ws_tokens) - 1:
                ambiguous_sentence += '.'

            # Capitalize sentence beginning
            for s in [filled_sentence, ambiguous_sentence]:
                if s[0] == s[0].lower():
                    s = s.capitalize()
                # Add periods to sentence ends
                if s[-1] not in string.punctuation:
                    s += '.'

            sample_features.append({'sample': sample,
                                    'init_gap_id': init_gap_id,
                                    'referent': referent,
                                    'filled_sentence': filled_sentence,
                                    'filled_spacy_doc': doc,
                                    'ambiguous_sentence': ambiguous_sentence,
                                    'ws_gap_ids': ws_gap_ids,
                                    'spacy_gap_ids': spacy_gap_ids,
                                    'ws_tokens': ws_tokens,
                                    'ws_tokens_lower': ws_tokens_lower,
                                    'spacy_tokens': spacy_tokens,
                                    'spacy_tokens_lower': [t.lower() for t in spacy_tokens],
                                    'ptb_tags': ptb_tags,
                                    'pos_tags': pos_tags,
                                    'dep_tags': dep_tags,
                                    'morph_features': morph_features,
                                    'ws_spacy_map': ws_spacy_map,
                                    'spacy_ws_map': spacy_ws_map,
                                    'ws_noun_tokens': ws_noun_tokens,
                                    'ws_noun_positions': ws_noun_loc,
                                    'ws_option1_position': ws_option1_loc,
                                    'ws_option2_position': ws_option2_loc,
                                    'deleted_det': deleted_det,
                                    'referent_is_true': is_true})

    return sample_features  # returns 4 items (2x correct filler + 2 x incorrect filler)



class S2SInputExample(object):
    def __init__(self, guid, src_sentence, tgt_sentence, filler=None):
        self.guid = guid
        self.filler = filler
        self.src_sentence = src_sentence
        self.tgt_sentence = tgt_sentence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, src_ids, src_mask, src_segments, tgt_ids, tgt_mask):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.src_segments = src_segments
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask


class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets. """

    def __init__(self, args):
        self.args = args

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        raise NotImplementedError()

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        raise NotImplementedError()

    @classmethod
    def _read_jsonl(cls, input_file):
        """ Reads a .jsonl file. """
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records


class WinoFullProcessor(DataProcessor):

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

    def _create_examples(self, records):
        # Works for monolingual training / evaluation
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            src_sentence = record['sentence']
            true_filler = record['option1'] if int(record['answer']) == 1 else record['option2']
            tgt_sentence = src_sentence.replace('_', true_filler)

            example = S2SInputExample(guid=guid,
                                      filler=true_filler,
                                      src_sentence=src_sentence,
                                      tgt_sentence=tgt_sentence)
            examples.append(example)

        return examples


class WinoXMTProcessor(DataProcessor):

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

    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            src_sentence = record['sentence']
            tgt_sentence = record['translation1'] if int(record['answer']) == 1 else record['translation2']

            example = S2SInputExample(guid=guid,
                                      src_sentence=src_sentence,
                                      tgt_sentence=tgt_sentence)
            examples.append(example)

        return examples


class WinoXMLMPronProcessor(DataProcessor):

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

    def _create_examples(self, records):
        # Works for monolingual training / evaluation
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            src_sentence = record['tgt_context']
            true_filler = record['option1'] if int(record['answer']) == 1 else record['option2']
            tgt_sentence = src_sentence.replace('_', true_filler)

            example = S2SInputExample(guid=guid,
                                      filler=true_filler,
                                      src_sentence=src_sentence,
                                      tgt_sentence=tgt_sentence)
            examples.append(example)

        return examples


class WinoXMLMWordsProcessor(DataProcessor):

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

    def _create_examples(self, records):
        # Works for monolingual and cross-lingual training / evaluation
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            src_sentence = record['context_{:s}'.format(self.args.src_lang)]
            true_filler = record['option1_{:s}'.format(self.args.tgt_lang)] if int(record['answer']) == 1 else \
                record['option2_{:s}'.format(self.args.tgt_lang)]
            tgt_sentence = src_sentence.replace('_', true_filler)

            example = S2SInputExample(guid=guid,
                                      filler=true_filler,
                                      src_sentence=src_sentence,
                                      tgt_sentence=tgt_sentence)
            examples.append(example)

        return examples


def convert_examples_to_features(examples,
                                 model_type,
                                 tokenizer,
                                 max_seq_len,
                                 pad_on_left=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    """ WinoGrande feature construction class. """
    # Determine maximum input tokens length
    max_src_length, max_tgt_length = 0, 0
    src_encodings = list()
    tgt_encodings = list()

    for ex_index, example in enumerate(examples):
        # Optionally truncate
        if max_seq_len > 0:
            example.src_sentence = example.src_sentence[:]

        if 'bart' in model_type:
            # Replace with a single mask token
            mask_sequence = tokenizer.mask_token
        else:
            num_masks = len(tokenizer.tokenize(example.filler))
            mask_sequence = ' '.join([tokenizer.mask_token] * num_masks)

        src_encoded = tokenizer(example.src_sentence.replace('_', mask_sequence))
        src_length = len(src_encoded['input_ids'])
        max_src_length = src_length if src_length > max_src_length else max_src_length
        src_encodings.append(src_encoded)

        with tokenizer.as_target_tokenizer():
            tgt_encoded = tokenizer(example.tgt_sentence)
            tgt_length = len(tgt_encoded['input_ids'])
            max_tgt_length = tgt_length if tgt_length > max_tgt_length else max_tgt_length
            tgt_encodings.append(tgt_encoded)

    max_pad_size = max(max_src_length, max_tgt_length)
    features = list()
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))

        # Process source sequence
        src_encoded = src_encodings[ex_index]
        src_ids = src_encoded['input_ids']
        src_tokens = tokenizer.convert_ids_to_tokens(src_ids)
        src_mask = src_encoded['attention_mask']
        src_segments = src_encoded.get('token_type_ids', [sequence_a_segment_id] * len(src_ids))

        # Zero-pad up to the sequence length
        src_padding_length = max_pad_size - len(src_ids)
        if pad_on_left:
            src_ids = ([pad_token] * src_padding_length) + src_ids
            src_mask = ([0 if mask_padding_with_zero else 1] * src_padding_length) + src_mask
            src_segments = ([pad_token_segment_id] * src_padding_length) + src_segments
        else:
            src_ids = src_ids + ([pad_token] * src_padding_length)
            src_mask = src_mask + ([0 if mask_padding_with_zero else 1] * src_padding_length)
            src_segments = src_segments + ([pad_token_segment_id] * src_padding_length)

        # Process target sequence
        with tokenizer.as_target_tokenizer():
            tgt_encoded = tgt_encodings[ex_index]
            tgt_ids = tgt_encoded['input_ids']
            tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids)
            tgt_mask = tgt_encoded['attention_mask']

        # Mask out padding positions, so that they don't contribute to the loss calculation
        tgt_padding_length = max_pad_size - len(tgt_ids)
        if pad_on_left:
            tgt_ids = ([-100] * tgt_padding_length) + tgt_ids
            tgt_mask = ([0 if mask_padding_with_zero else 1] * tgt_padding_length) + tgt_mask
        else:
            tgt_ids = tgt_ids + ([-100] * tgt_padding_length)
            tgt_mask = tgt_mask + ([0 if mask_padding_with_zero else 1] * tgt_padding_length)

        features.append(
            InputFeatures(src_ids=src_ids,
                          src_mask=src_mask,
                          src_segments=src_segments,
                          tgt_ids=tgt_ids,
                          tgt_mask=tgt_mask))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def set_seed(args):
    """ Sets the seed to support reproducibility. """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    """ Keep a maximum of args.save_total_limit checkpoints. """
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


def get_token_loss(args, lm_logits, target_ids, target_mask, model_type=None):
    """ Compute token-level loss per batch during evaluation. """
    # Declare loss function
    loss_fct = CrossEntropyLoss(reduction='none')
    if model_type is None:
        model_type = args.model_type

    if model_type in ['t5', 'bart']:
        # Obtain logits to compute token-level loss / perplexity
        batch_size, max_length, vocab_size = lm_logits.shape

        # Compute loss for each instance and each token
        lm_logits = lm_logits.view(-1, vocab_size)
        lm_labels = target_ids[:, 1:].clone().contiguous().view(-1)
        token_loss = loss_fct(lm_logits, lm_labels).view(batch_size, max_length)

    else:
        # Obtain logits to compute token-level loss / perplexity
        shift_logits = lm_logits[..., :-1, :].contiguous()
        batch_size, max_length, vocab_size = shift_logits.shape

        # Compute loss for each instance and each token
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = target_ids[..., 1:].contiguous().view(-1)
        token_loss = loss_fct(shift_logits, shift_labels).view(batch_size, max_length)

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


def compute_gen_metrics(preds, targets):
    """ Aggregates generation metrics. """
    assert len(preds) == len(targets)
    return {'BLEU-4': str(compute_bleu(preds, targets))}
