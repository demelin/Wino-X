import re
import os
import json
import spacy
import string
import logging
import argparse
import language_tool_python

import numpy as np

from test_animacy import test_if_animate
from spacy_tag_map import TAG_MAP
from util import read_jsonl, parse_sample_pair

CONNECTORS = [' — ', ', because ', ', so ', ', but ']


def _check_grammaticality(sentence):
    """ Evaluates whether the input sentence contains grammar errors. """
    matches = tool.check(sentence)
    num_errors = 0
    error_types = dict()
    # Count significant errors
    for m in matches:
        if m.ruleId not in ['UPPERCASE_SENTENCE_START', 'PROFANITY', 'COMMA_PARENTHESIS_WHITESPACE',
                            'COMMA_COMPOUND_SENTENCE', 'PRP_COMMA', 'IT_IS_JJ_TO_VBG', 'NON3PRS_VERB']:
            num_errors += 1
    # Document error types
    for m in matches:
        m_type = m.ruleId
        if error_types.get(m_type, None) is None:
            error_types[m_type] = 1
        else:
            error_types[m_type] += 1
    return num_errors, error_types


def annotate_samples(json_path, return_filtered, exclude_multi_sentence_samples):
    """ Annotates WinoGrande samples with linguistic information that can be used for subsequent filtering. """

    # Read
    samples = read_jsonl(json_path)

    # Pair
    paired_samples = dict()
    for s in samples:
        s_id = '-'.join(s['qID'].split('-')[:-1])
        if not paired_samples.get(s_id, None):
            paired_samples[s_id] = [s]
        else:
            paired_samples[s_id].append(s)

    # Filter (first pass, simple heuristics)
    first_filter_pass = {'kept': dict(),
                         'overlap': dict(),
                         'sentence_final': dict(),
                         'activities': dict(),
                         'multi_sentence': dict()}
    for s_id in paired_samples.keys():
        pair = paired_samples[s_id]
        keep_pair = True
        sentence1 = pair[0]['sentence'].strip()
        sentence2 = pair[1]['sentence'].strip()

        # Check for referent overlap
        option1 = pair[0]['option1']
        option2 = pair[0]['option2']
        option1_tokens = re.findall(r"[\w']+", option1)
        option2_tokens = re.findall(r"[\w']+", option2)
        option1_tokens_lower = re.findall(r"[\w']+", option1.lower())
        option2_tokens_lower = re.findall(r"[\w']+", option2.lower())
        if len(option1_tokens) > 1 or len(option2_tokens) > 1:
            if option1_tokens[-1] == option2_tokens[-1]:
                first_filter_pass['overlap'][s_id] = pair
                keep_pair = False
                logging.warning('=' * 20)
                logging.warning('Overlap between referents:')
                logging.warning('Option 1: {}'.format(option1))
                logging.warning('Option 2: {}'.format(option2))

        if keep_pair:
            # Check gap position
            sent1_tokens = pair[0]['sentence'].strip().lower().translate(
                str.maketrans('', '', string.punctuation.replace('_', ''))).split()
            sent1_tokens = [t.strip() for t in sent1_tokens if len(t.strip()) > 0]
            sent2_tokens = pair[1]['sentence'].strip().lower().translate(
                str.maketrans('', '', string.punctuation.replace('_', ''))).split()
            sent2_tokens = [t.strip() for t in sent2_tokens if len(t.strip()) > 0]
            if sent1_tokens[-1] == '_' or sent2_tokens[-1] == '_':
                first_filter_pass['sentence_final'][s_id] = (pair, 'sentence_final')
                # pair_tags.append('sentence_final')
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Gap is sentence-final:')
                logging.warning('Sentence 1: {}'.format(sentence1))
                logging.warning('Sentence 2: {}'.format(sentence2))

        if keep_pair:
            if option1_tokens_lower[-1].endswith('ing') and option2_tokens_lower[-1].endswith('ing'):
                first_filter_pass['activities'][s_id] = pair
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Referents are activities:')
                logging.warning('Option 1: {}'.format(option1))
                logging.warning('Option 2: {}'.format(option2))

        if keep_pair:
            sent1_parts = len([p for p in pair[0]['sentence'].strip().split('.') if len(p.strip()) > 0])
            sent2_parts = len([p for p in pair[1]['sentence'].strip().split('.') if len(p.strip()) > 0])
            if sent1_parts > 2 or sent2_parts > 2:
                first_filter_pass['multi_sentence'][s_id] = pair
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Sample consists of more than two sentences:')
                logging.warning('Sentence 1: {}'.format(sentence1))
                logging.warning('Sentence 2: {}'.format(sentence2))

        if keep_pair:
            first_filter_pass['kept'][s_id] = pair

    # Report
    logging.warning('-' * 20)
    logging.warning(
        'FIRST PASS COMPLETED! Annotated and evaluated {:d} WinoGrande samples'.format(len(paired_samples.keys())))
    logging.warning('Samples kept: {:d}'.format(len(first_filter_pass['kept'])))
    logging.warning('Samples dropped: {:d}'.format(len(paired_samples.keys()) - len(first_filter_pass['kept'])))

    # Filter (second pass, linguistic features)
    # 1. Referents do not agree in number
    # 2. Ambiguous pronoun has a case other than nominative
    # 3. Referents are part of a compound noun
    # 4. Gap is preceded by a preposition or adjective
    # 5. Ambiguous pronoun replacement introduces grammatical errors
    paired_samples = first_filter_pass['kept']
    second_filter_pass = {'kept': dict(),
                          'animate_ref': dict(),
                          'animal_ref': dict(),
                          'referent_not_in_sentence': dict(),
                          'illegal_referents': dict(),
                          'no_number_agreement': dict(),
                          'inflected_pronoun': dict(),
                          'part_of_compound': dict(),
                          'adj_before_gap': dict(),
                          'referent_mention_in_compound': dict(),
                          'referent_mention_in_phrase': dict(),
                          'referent_has_conj_head': dict(),
                          'large_score_diff': dict(),
                          'ungrammatical': dict()}

    kept_inflected_pairs = 0
    for s_iid, s_id in enumerate(paired_samples.keys()):
        keep_pair = True
        pair = paired_samples[s_id]
        pair_tags = list()
        option1 = pair[0]['option1'].lower()
        option2 = pair[0]['option2'].lower()
        option1_tokens = [t for t in option1.lower().split() if len(t) > 0]
        option2_tokens = [t for t in option2.lower().split() if len(t) > 0]
        # Process samples
        sample_features = parse_sample_pair(pair, nlp, scorer, CONNECTORS, exclude_multi_sentence_samples)
        if sample_features is None:
            logging.warning('=== SKIPPED ===')
            logging.warning('{}'.format(pair))
            continue
            
        s1_sentence = sample_features[0]['filled_sentence']
        s2_sentence = sample_features[1]['filled_sentence']
        s3_sentence = sample_features[2]['filled_sentence']
        s4_sentence = sample_features[3]['filled_sentence']
        s1_ambiguous_sentence = sample_features[0]['ambiguous_sentence']
        s2_ambiguous_sentence = sample_features[1]['ambiguous_sentence']
        s3_ambiguous_sentence = sample_features[2]['ambiguous_sentence']
        s4_ambiguous_sentence = sample_features[3]['ambiguous_sentence']

        logging.warning('+' * 20)
        logging.warning('Checking new pair ...')
        logging.warning('{}, {}'.format(s1_sentence, sample_features[0]['referent']))
        logging.warning(s1_ambiguous_sentence)
        logging.warning('-' * 5)
        logging.warning('{}, {}'.format(s2_sentence, sample_features[1]['referent']))
        logging.warning(s2_ambiguous_sentence)
        logging.warning('-' * 5)
        logging.warning('-' * 5)
        logging.warning('{}, {}'.format(s3_sentence, sample_features[2]['referent']))
        logging.warning(s3_ambiguous_sentence)
        logging.warning('-' * 5)
        logging.warning('{}, {}'.format(s4_sentence, sample_features[3]['referent']))
        logging.warning(s4_ambiguous_sentence)

        logging.warning('+' * 20)

        # Get number information
        s1_spacy_target_id = sample_features[0]['spacy_gap_ids'][-1]
        s1_ptb_tags = sample_features[0]['ptb_tags']
        r1_tag = s1_ptb_tags[s1_spacy_target_id]
        s2_spacy_target_id = sample_features[1]['spacy_gap_ids'][-1]
        s2_ptb_tags = sample_features[1]['ptb_tags']
        r2_tag = s2_ptb_tags[s2_spacy_target_id]
        r1_is_singular = TAG_MAP[r1_tag].get('Number_sing', False)
        r1_is_plural = TAG_MAP[r1_tag].get('Number_plur', False)
        r2_is_singular = TAG_MAP[r2_tag].get('Number_sing', False)
        r2_is_plural = TAG_MAP[r2_tag].get('Number_plur', False)

        if keep_pair:
            if len(sample_features[0]['ws_option1_position']) < len(option1_tokens) or \
                    len(sample_features[0]['ws_option2_position']) < len(option2_tokens) or \
                    len(sample_features[1]['ws_option1_position']) < len(option1_tokens) or \
                    len(sample_features[1]['ws_option2_position']) < len(option2_tokens):
                second_filter_pass['referent_not_in_sentence'][s_id] = pair[:2] + [dict()]  # for consistency
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Referent not in sentence:')
                logging.warning('Sentence 1 tokens: {} | ref1: {} | ref2: {}'
                                .format(s1_ambiguous_sentence, option1, option2))
                logging.warning('Sentence 2 tokens: {} | ref1: {} | ref2: {}'
                                .format(s2_ambiguous_sentence, option1, option2))

        if keep_pair:
            # Check if referents contain words that are not adjectives or nouns
            # 'VERB' is included to counteract tagging errors
            ref1_pos = list()
            for gid in sample_features[0]['spacy_gap_ids']:
                pos = sample_features[0]['pos_tags'][gid]
                ref1_pos.append(pos)
                if pos not in ['NOUN', 'PROPN', 'ADJ', 'VERB']:
                    keep_pair = False
            ref2_pos = list()
            for gid in sample_features[1]['spacy_gap_ids']:
                pos = sample_features[1]['pos_tags'][gid]
                ref2_pos.append(pos)
                if pos not in ['NOUN', 'PROPN', 'ADJ', 'VERB']:
                    keep_pair = False

            if not keep_pair:
                extra_info = {'ref1_pos': ref1_pos,
                              'ref2_pos': ref2_pos}
                second_filter_pass['illegal_referents'][s_id] = pair[:2] + [extra_info]

                logging.warning('=' * 20)
                logging.warning('Illegal referents:')
                logging.warning('Sentence 1: {} | target POS: {}'.format(s1_sentence, ref1_pos))
                logging.warning('Sentence 2: {} | target POS: {}'.format(s2_sentence, ref2_pos))

        if keep_pair:
            # Check if referents agree in number
            if ((option1.endswith('s') and not (option1.endswith('ss'))) and not
                (option2.endswith('s') and not (option2.endswith('ss')))) or \
                    ((option2.endswith('s') and not (option2.endswith('ss'))) and not
                        (option1.endswith('s') and not (option1.endswith('ss')))):
                # This also removes non-noun referents
                if (r1_is_singular and not r2_is_singular) or (r1_is_plural and not r2_is_plural) or \
                        not (r1_is_singular or r1_is_plural) or not (r2_is_singular or r2_is_plural):

                    extra_info = {'r1_is_singular': r1_is_singular,
                                  'r1_is_plural': r1_is_plural,
                                  'r2_is_singular': r2_is_singular,
                                  'r2_is_plural': r2_is_plural}
                    second_filter_pass['no_number_agreement'][s_id] = pair[:2] + [extra_info]
                    keep_pair = False

                    logging.warning('=' * 20)
                    logging.warning('No number agreement:')
                    logging.warning('Sentence 1: {} | is singular: {} | is plural: {}'
                                    .format(s1_sentence, r1_is_singular, r1_is_plural))
                    logging.warning('Sentence 2: {} | is singular: {} | is plural: {}'
                                    .format(s2_sentence, r2_is_singular, r2_is_plural))

        if keep_pair:
            # Check if the filled-in referent functions as a subject within the 'gap-clause'
            subj1_found = False
            r1_dep_tags = list()
            for gid in sample_features[0]['spacy_gap_ids']:
                r1_dep_tag = sample_features[0]['dep_tags'][gid]
                r1_dep_tags.append(r1_dep_tag)
                if 'subj' in r1_dep_tag:
                    subj1_found = True
            subj2_found = False
            r2_dep_tags = list()
            for gid in sample_features[1]['spacy_gap_ids']:
                r2_dep_tag = sample_features[1]['dep_tags'][gid]
                r2_dep_tags.append(r2_dep_tag)
                if 'subj' in r2_dep_tag:
                    subj2_found = True
            if not subj1_found or not subj2_found:
                pair_tags.append('inflected_pronoun')

                logging.warning('=' * 20)
                logging.warning('Pronoun is inflected:')
                logging.warning(
                    'Sentence 1: {} | referent DEP tags: {}'.format(s1_sentence, r1_dep_tags))
                logging.warning(
                    'Sentence 2: {} | referent DEP tags: {}'.format(s2_sentence, r2_dep_tags))

        if keep_pair:
            # Check if either referent is likely part of a compound noun (or phrase)
            r1_right_context_id = sample_features[0]['spacy_gap_ids'][-1] + 1
            r1_right_context_tag = sample_features[0]['pos_tags'][r1_right_context_id]
            r1_right_context_token = sample_features[0]['spacy_tokens'][r1_right_context_id]
            r2_right_context_id = sample_features[1]['spacy_gap_ids'][-1] + 1
            r2_right_context_tag = sample_features[1]['pos_tags'][r2_right_context_id]
            r2_right_context_token = sample_features[1]['spacy_tokens'][r2_right_context_id]
            if r1_right_context_tag in ['NOUN', 'PROPN'] or r2_right_context_tag in ['NOUN', 'PROPN'] or \
                    r1_right_context_token.endswith('ing') or r2_right_context_token.endswith('ing') or \
                    r1_right_context_token.startswith('\'') or r2_right_context_token.startswith('\'') or \
                    r1_right_context_token in ['one', 'ones'] or r2_right_context_token in ['one', 'ones']:
                extra_info = {'r1_right_context_token': r1_right_context_token,
                              'r1_right_context_pos': r1_right_context_tag,
                              'r2_right_context_token': r2_right_context_token,
                              'r2_right_context_pos': r2_right_context_tag}
                second_filter_pass['part_of_compound'][s_id] = pair[:2] + [extra_info]
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Gap is part of compound:')
                logging.warning('Sentence 1: {} | right context token: {} | right context POS: {}'
                                .format(s1_sentence, r1_right_context_token, r1_right_context_tag))
                logging.warning('Sentence 2: {} | right context token: {} | right context POS: {}'
                                .format(s2_sentence, r2_right_context_token, r2_right_context_tag))

        if keep_pair:
            # Check if either gap is preceded by an adjective
            r1_left_context_id = sample_features[0]['spacy_gap_ids'][0] - 1
            r1_left_context_token = sample_features[0]['spacy_tokens'][r1_left_context_id]
            r1_left_context_tag = sample_features[0]['pos_tags'][r1_left_context_id]
            r2_left_context_id = sample_features[1]['spacy_gap_ids'][0] - 1
            r2_left_context_token = sample_features[1]['spacy_tokens'][r2_left_context_id]
            r2_left_context_tag = sample_features[1]['pos_tags'][r2_left_context_id]
            if r1_left_context_tag == 'ADJ' or r2_left_context_tag == 'ADJ':
                extra_info = {'r1_left_context_token': r1_left_context_token,
                              'r1_left_context_pos': r1_left_context_tag,
                              'r2_left_context_token': r2_left_context_token,
                              'r2_left_context_pos': r2_left_context_tag}
                second_filter_pass['adj_before_gap'][s_id] = pair[:2] + [extra_info]
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Adjective before gap:')
                logging.warning('Sentence 1: {} | left context token: {} | left context POS: {}'
                                .format(s1_sentence, r1_left_context_token, r1_left_context_tag))
                logging.warning('Sentence 2: {} | right context token: {} | right context POS: {}'
                                .format(s2_sentence, r2_left_context_token, r2_left_context_tag))

        if keep_pair:
            # Check if either referent mention in both sentences is part of a compound
            ref1 = sample_features[0]['referent'].strip().lower().split()[-1]
            ref2 = sample_features[1]['referent'].strip().lower().split()[-1]
            s1_spacy_doc = sample_features[0]['filled_spacy_doc']
            s2_spacy_doc = sample_features[1]['filled_spacy_doc']
            s1_noun_chunks = [ch for ch in s1_spacy_doc.noun_chunks]
            s2_noun_chunks = [ch for ch in s2_spacy_doc.noun_chunks]
            s1_chunk_tokens = list()
            s2_chunk_tokens = list()
            for chunks, tokens in [(s1_noun_chunks, s1_chunk_tokens), (s2_noun_chunks, s2_chunk_tokens)]:
                for ch in chunks:
                    ch_tokens = list()
                    for t in ch:
                        ch_tokens.append(t.text)
                    tokens.append(ch_tokens)

            for chunks in [s1_noun_chunks, s2_noun_chunks]:
                for ch in chunks:
                    ch_tok = [t.text for t in ch]
                    ch_pos_tags = [t.pos_ for t in ch]
                    for t_id, t in enumerate(ch_tok[:-1]):
                        if t in [ref1, ref2]:
                            try:
                                if ch_pos_tags[t_id + 1] in ['NOUN', 'PROPN']:
                                    keep_pair = False
                                    break
                            except IndexError:
                                continue

            if not keep_pair:
                extra_info = {'s1_noun_chunks': s1_chunk_tokens,
                              's2_noun_chunks': s2_chunk_tokens}
                second_filter_pass['referent_mention_in_compound'][s_id] = pair[:2] + [extra_info]

                logging.warning('=' * 20)
                logging.warning('Referent is part of compound:')
                logging.warning('Sentence 1: {} | noun chunks: {}'
                                .format(s1_sentence, s1_noun_chunks))
                logging.warning('Sentence 2: {} | noun chunks: {}'
                                .format(s2_sentence, s2_noun_chunks))
        if keep_pair:
            # Check if both sentences are grammatical
            s1_num_gramm_errors, s1_gramm_error_types = _check_grammaticality(s1_sentence)
            s2_num_gramm_errors, s2_gramm_error_types = _check_grammaticality(s2_sentence)

            s1a_num_gramm_errors, s1a_gramm_error_types = _check_grammaticality(s1_ambiguous_sentence)
            s2a_num_gramm_errors, s2a_gramm_error_types = _check_grammaticality(s2_ambiguous_sentence)

            if s1a_num_gramm_errors > s1_num_gramm_errors or s2a_num_gramm_errors > s2_num_gramm_errors:
                extra_info = {'s1_grammar_error_types': s1_gramm_error_types,
                              's2_grammar_error_types': s2_gramm_error_types,
                              's1a_grammar_error_types': s1a_gramm_error_types,
                              's2a_grammar_error_types': s2a_gramm_error_types}
                second_filter_pass['ungrammatical'][s_id] = pair[:2] + [extra_info]
                keep_pair = False

                logging.warning('=' * 20)
                logging.warning('Modified sample is ungrammatical:')
                logging.warning('Full sentence 1: {} | # grammar errors: {} | error types: {}'
                                .format(s1_sentence, s1_num_gramm_errors, s1_gramm_error_types))
                logging.warning('Ambiguous sentence 1: {} | # grammar errors: {} | error types: {}'
                                .format(s1_ambiguous_sentence, s1a_num_gramm_errors, s1a_gramm_error_types))
                logging.warning('Full sentence 2: {} | # grammar errors: {} | error types: {}'
                                .format(s2_sentence, s2_num_gramm_errors, s2_gramm_error_types))
                logging.warning('Ambiguous sentence 2: {} | # grammar errors: {} | error types: {}'
                                .format(s2_ambiguous_sentence, s2a_num_gramm_errors, s2a_gramm_error_types))

        if keep_pair:

            sample1, sample2 = pair
            sample3 = {k: v for k, v in sample1.items()}
            sample4 = {k: v for k, v in sample2.items()}

            for s_num, sample in enumerate([sample1, sample2, sample3, sample4]):
                sample['ambiguous_sentence'] = sample_features[s_num]['ambiguous_sentence']
                sample['it_id'] = sample_features[s_num]['init_gap_id']
                sample['referent'] = sample_features[s_num]['referent']
                sample['referent_is_true'] = sample_features[s_num]['referent_is_true']
                sample['filled_sentence'] = sample_features[s_num]['filled_sentence']
                sample['ws_gap_ids'] = sample_features[s_num]['ws_gap_ids']
                sample['spacy_gap_ids'] = sample_features[s_num]['spacy_gap_ids']
                sample['spacy_tokens'] = sample_features[s_num]['spacy_tokens']
                sample['ptb_tags'] = sample_features[s_num]['ptb_tags']
                sample['pos_tags'] = sample_features[s_num]['pos_tags']
                sample['morph_features'] = sample_features[s_num]['morph_features']
                sample['ws_spacy_map'] = sample_features[s_num]['ws_spacy_map']
                sample['spacy_ws_map'] = sample_features[s_num]['spacy_ws_map']
                sample['is_multi_sentence'] = len([s for s in sample['sentence'].split('.') if len(s) > 0]) > 1
                sample['ws_noun_tokens'] = sample_features[s_num]['ws_noun_tokens']
                sample['ws_noun_positions'] = sample_features[s_num]['ws_noun_positions']
                sample['ws_option1_position'] = sample_features[s_num]['ws_option1_position']
                sample['ws_option2_position'] = sample_features[s_num]['ws_option2_position']

            sample1['referent1_is_singular'] = sample2['referent1_is_singular'] = sample3['referent1_is_singular'] = \
                sample4['referent1_is_singular'] = r1_is_singular
            sample1['referent2_is_singular'] = sample2['referent2_is_singular'] = sample3['referent2_is_singular'] = \
                sample4['referent2_is_singular'] = r2_is_singular
            sample1['referent1_is_inflected'] = sample2['referent1_is_inflected'] = sample3['referent1_is_inflected'] = \
                sample4['referent1_is_inflected'] = len(pair_tags) > 0
            sample1['referent2_is_inflected'] = sample2['referent2_is_inflected'] = sample3['referent2_is_inflected'] = \
                sample4['referent2_is_inflected'] = len(pair_tags) > 0
            
            second_filter_pass['kept'][s_id] = [sample1, sample2, sample3, sample4]
            kept_inflected_pairs += int(sample1['referent1_is_inflected'] or sample1['referent2_is_inflected'])

        # Report periodically
        if s_iid > 0 and (s_iid + 1) % 100 == 0:
            logging.warning('\n\n')
            logging.warning('-' * 20)
            logging.warning('SECOND PASS: Annotated and evaluated {:d} WinoGrande samples'.format(s_iid + 1))
            logging.warning('Samples kept: {:d}'.format(len(second_filter_pass['kept'])))
            logging.warning('Samples dropped: {:d}\n\n'.format((s_iid + 1) - len(second_filter_pass['kept'])))
            logging.warning('Kept inflected pairs (samples): {:d} ({:d})'.format(kept_inflected_pairs,
                                                                                 kept_inflected_pairs * 2))

    # Write results to files
    if return_filtered:

        for category in ['overlap', 'sentence_final', 'activities', 'multi_sentence']:
            logging.warning('=' * 20)
            logging.warning('Filtered-out {:d} samples due to {:s}'
                            .format(len(first_filter_pass['{:s}'.format(category)]), category))
            out_path = json_path[:-6] + '_{:s}.jsonl'.format(category)
            with open(out_path, 'w', encoding='utf8') as out_f:
                for s_iid, s_id in enumerate(first_filter_pass['{:s}'.format(category)].keys()):
                    sample1, sample2 = first_filter_pass['{:s}'.format(category)][s_id]
                    if s_iid > 0:
                        out_f.write('\n')
                    out_f.write(json.dumps(sample1))
                    out_f.write('\n')
                    out_f.write(json.dumps(sample2))

        for category in ['animate_ref', 'referent_not_in_sentence', 'illegal_referents', 'no_number_agreement',
                         'part_of_compound', 'referent_mention_in_compound', 'referent_has_conj_head',
                         'referent_mention_in_phrase', 'adj_before_gap', 'ungrammatical']:
            logging.warning('=' * 20)
            logging.warning('Filtered-out {:d} samples due to {:s}'
                            .format(len(second_filter_pass['{:s}'.format(category)]), category))
            out_path = json_path[:-6] + '_{:s}.jsonl'.format(category)
            with open(out_path, 'w', encoding='utf8') as out_f:
                for s_iid, s_id in enumerate(second_filter_pass['{:s}'.format(category)].keys()):
                    sample1, sample2, extra_info = second_filter_pass['{:s}'.format(category)][s_id]
                    if s_iid > 0:
                        out_f.write('\n')
                    out_f.write(json.dumps(sample1))
                    out_f.write('\n')
                    out_f.write(json.dumps(sample2))
                    if len(extra_info.keys()) > 0:
                        out_f.write('\n')
                        out_f.write(json.dumps(extra_info))

    logging.warning('Kept {:d} samples overall'.format(len(second_filter_pass['kept']) * 2))
    out_path = json_path[:-6] + '_filtered_dataset.jsonl'
    with open(out_path, 'w', encoding='utf8') as out_f:
        for s_iid, s_id in enumerate(second_filter_pass['kept'].keys()):
            if s_iid > 0:
                out_f.write('\n')
            for i_id, item in enumerate(second_filter_pass['kept'][s_id]):
                out_f.write(json.dumps(item))
                if i_id < 3:
                    out_f.write('\n')

    logging.warning('Saved filtered-out samples to their respective destinations.')
    logging.warning('-' * 20)
    num_kept_multi_sentence = 0
    for k in second_filter_pass['kept']:
        for s in second_filter_pass['kept'][k]:
            if s['is_multi_sentence']:
                num_kept_multi_sentence += 1
    logging.warning('Of {:d} samples kept, {:d} are multi-sentence'
                    .format(len(second_filter_pass['kept']), num_kept_multi_sentence))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the JSON file containing the WinoGrande data')
    parser.add_argument('--return_filtered', action='store_true',
                        help='whether to return filtered-out samples')
    parser.add_argument('--exclude_multi_sentence_samples', action='store_true',
                        help='whether to exclude multi-sentence samples from the derived datasets; recommended')
    args = parser.parse_args()

    # Initialize language tool
    tool = language_tool_python.LanguageTool('en-US')
    # Load spaCy model
    nlp = spacy.load('en_core_web_lg')
    # Setup logging (NOTE: WARNING is used instead of INFO so as to avoid overly verbose MLM scorer output)
    base_dir = '/'.join(args.data_path.split('/')[:-1])
    file_name = args.data_path.split('/')[-1][:-6] + '_annotation'
    log_dir = '{:s}/logs/'.format(base_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = '{:s}{:s}.log'.format(log_dir, file_name)
    logger = logging.getLogger('')
    logger.setLevel(logging.WARNING)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    scorer = None
    if not args.exclude_multi_sentence_samples:
        import mxnet as mx
        from mlm.scorers import MLMScorer
        from mlm.models import get_pretrained

        ctxs = [mx.gpu(0)]
        model, vocab, tokenizer = get_pretrained(ctxs, 'roberta-large-en-cased')
        scorer = MLMScorer(model, vocab, tokenizer, ctxs)

    annotate_samples(args.data_path, args.return_filtered, args.exclude_multi_sentence_samples)
