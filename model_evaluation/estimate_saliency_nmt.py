import os
import json
import math
import torch
import random
import fastBPE
import argparse

import numpy as np

from Bio import pairwise2
from functools import reduce
from fairseq import checkpoint_utils
from sacremoses import MosesTokenizer
from subword_nmt.apply_bpe import BPE, read_vocabulary

from pingouin import ttest

from eval_util import read_jsonl, get_final_index, get_sublist_indices, segment_seq, prepare_inputs, compute_perplexity,\
    plot_saliency_map


def _get_gradient_norm_scores(src_sent, tgt_sent, target_token):
    """ Computes the L2 norm of the gradients of each source embedding with respect to the target pronoun. """

    # Preprocess sequence
    src_sent_bpe, src_sent_ids = \
        segment_seq(src_sent, src_mt, src_bpe_model, model.encoder.dictionary, args.use_subword_nmt)
    tgt_sent_bpe, tgt_sent_ids = \
        segment_seq(tgt_sent, tgt_mt, tgt_bpe_model, model.decoder.dictionary, args.use_subword_nmt)

    # Identify source pronoun ID and its position in target sequence
    try:
        src_dict_id = model.encoder.dictionary.index('it')
        src_idx = get_final_index(src_sent_ids.cpu().numpy().tolist(), src_dict_id)
    except AssertionError:
        src_dict_id = model.encoder.dictionary.index('It')
        src_idx = get_final_index(src_sent_ids.cpu().numpy().tolist(), src_dict_id)

    # Identify target pronoun ID and its position in target sequence
    tgt_dict_id = model.decoder.dictionary.index(target_token)
    tgt_idx = get_final_index(tgt_sent_ids.cpu().numpy().tolist(), tgt_dict_id)

    # Perform a full forward pass for the true target
    model.zero_grad()
    sample = prepare_inputs(src_sent_ids, tgt_sent_ids, model, task, device)
    loss, _, _ = criterion.forward_at_position(model, sample, tgt_idx)
    loss.backward()  # compute gradients w.r.t target pronoun translation probability
    embedding_grads = model.encoder.embed_tokens.weight.grad

    saliency_scores = list()
    # Select relevant embedding entries and compute L2 norm
    for tok_id, src_tok_idx in enumerate(sample['net_input']['src_tokens'][0]):
        grad_norm = torch.norm(embedding_grads[src_tok_idx, :], p=1, dim=0).detach().cpu().numpy().tolist()
        saliency_scores.append([model.encoder.dictionary[src_tok_idx], grad_norm])
    return saliency_scores, src_idx


def _get_prediction_diff_scores(src_sent, true_tgt_sent, false_tgt_sent, target_token):
    """ Iteratively masks source tokens and evaluates the probability of the target token of interest conditioned on
    the masked source """

    # Preprocess sequence
    src_sent_bpe, src_sent_ids = \
        segment_seq(src_sent, src_mt, src_bpe_model, model.encoder.dictionary, args.use_subword_nmt)
    true_tgt_sent_bpe, true_tgt_sent_ids = \
        segment_seq(true_tgt_sent, tgt_mt, tgt_bpe_model, model.decoder.dictionary, args.use_subword_nmt)
    false_tgt_sent_bpe, false_tgt_sent_ids = \
        segment_seq(false_tgt_sent, tgt_mt, tgt_bpe_model, model.decoder.dictionary, args.use_subword_nmt)

    # Check if the pronoun is segmented or not
    prn_bpe, prn_ids = \
        segment_seq(target_token, tgt_mt, tgt_bpe_model, model.decoder.dictionary, args.use_subword_nmt)

    if len(prn_ids) == 1:
        prn_ids = None
    else:
        # remove EOS
        prn_ids = prn_ids[:-1]
        prn_ids = prn_ids.detach().cpu().numpy().tolist()

    # Identify source pronoun ID and its position in source sequence
    try:
        src_dict_id = model.encoder.dictionary.index('it')
        src_idx = get_final_index(src_sent_ids.cpu().numpy().tolist(), src_dict_id)
    except AssertionError:
        src_dict_id = model.encoder.dictionary.index('It')
        src_idx = get_final_index(src_sent_ids.cpu().numpy().tolist(), src_dict_id)

    if type(prn_ids) == list:
        tgt_idx = get_sublist_indices(true_tgt_sent_ids.cpu().numpy().tolist(), prn_ids)[-1]
    else:
        try:
            # Identify target pronoun ID and its position in target sequence
            tgt_dict_id = model.decoder.dictionary.index(target_token)
            tgt_idx = [get_final_index(true_tgt_sent_ids.cpu().numpy().tolist(), tgt_dict_id)]
        except AssertionError:
            return None, None, None

    # Perform a full forward pass for the true target
    true_sample = prepare_inputs(src_sent_ids, true_tgt_sent_ids, model, task, device)
    full_src = true_sample['net_input']['src_tokens'].clone()

    with torch.no_grad():
        _, _, log_output = criterion(model, true_sample, reduce=False)
    true_model_probabilities = log_output['target_probabilities']

    assert log_output['targets'][0].detach().cpu().numpy().tolist() == \
           true_tgt_sent_ids.detach().cpu().numpy().tolist(), 'Target mismatch!'
    true_ppl = compute_perplexity(true_model_probabilities)

    ref_tok_probs = list()
    for ti in tgt_idx:
        ref_tok_probs.append(true_model_probabilities[ti])
    if len(ref_tok_probs) == 1:
        ref_tok_prob = ref_tok_probs[0]
    else:
        ref_tok_prob = reduce(lambda x, y: x * y, ref_tok_probs, 1)

    # Perform a full forward pass for the false target
    false_sample = prepare_inputs(src_sent_ids, false_tgt_sent_ids, model, task, device)
    with torch.no_grad():
        _, _, log_output = criterion(model, false_sample, reduce=False)
    false_model_probabilities = log_output['target_probabilities']
    false_ppl = compute_perplexity(false_model_probabilities)

    model_is_correct = bool(true_ppl < false_ppl)
    # Compute token-wise saliency scores
    saliency_scores = list()
    for tok_id, src_tok_idx in enumerate(true_sample['net_input']['src_tokens'][0]):
        # Mask-out source token
        true_sample['net_input']['src_tokens'] = full_src.clone()
        true_sample['net_input']['src_tokens'][0][tok_id] = mask_index
        # Perform a forward-pass
        with torch.no_grad():
            _, _, log_output = criterion(model, true_sample, reduce=False)

        tok_probs = list()
        for ti in tgt_idx:
            tok_probs.append(log_output['target_probabilities'][ti])
        if len(tok_probs) == 1:
            tok_prob = tok_probs[0]
        else:
            tok_prob = reduce(lambda x, y: x * y, tok_probs, 1)

        # Store results
        saliency_scores.append([model.encoder.dictionary[src_tok_idx], ref_tok_prob - tok_prob])

    return model_is_correct, saliency_scores, src_idx


def compute_saliency_scores(json_challenge_path, out_dir, saliency_method, pd_saliency_table_path):
    """ Identifies source tokens that are salient to the pronoun choice on the target side. """

    # Read-in samples ('pronoun1' and 'pronoun2' denote the target pronouns)
    samples = read_jsonl(json_challenge_path)

    # Pair contrastive sample samples and select relevant entries
    sample_pairs = dict()
    for s in samples:
        qid = s['qID'].split('-')[-2]
        sid = int(s['qID'].split('-')[-1])
        if sample_pairs.get(qid, None) is None:
            sample_pairs[qid] = dict()
        sample_pairs[qid][sid] = s

    # Read in the prediction_diff evaluation results, if provided
    pd_saliency_table = None
    if pd_saliency_table_path is not None:
        with open(pd_saliency_table_path, 'r', encoding='utf8') as pdp:
            pd_saliency_table = json.load(pdp)

    # Iterate over samples
    saliency_table = dict()
    for qid_id, qid in enumerate(sample_pairs.keys()):

        print('Checking pair {:d}'.format(qid_id))
        if qid_id > 0 and (qid_id + 1) % 100 == 0:
            print('Analysed {:d} contrastive pairs'.format(qid_id + 1))

        saliency_table[qid] = dict()
        for sid in sample_pairs[qid].keys():
            saliency_table[qid][sid] = dict()
            saliency_entry = saliency_table[qid][sid]
            # Unpack
            sample = sample_pairs[qid][sid]
            src = sample['sentence']
            true_tgt = sample['translation1'] if sample['answer'] == 1 else sample['translation2']
            false_tgt = sample['translation2'] if sample['answer'] == 1 else sample['translation1']
            true_ref = sample['referent1_en'] if sample['answer'] == 1 else sample['referent2_en']
            false_ref = sample['referent2_en'] if sample['answer'] == 1 else sample['referent1_en']
            tgt_pron = sample['pronoun1'] if sample['answer'] == 1 else sample['pronoun2']

            # Get scores
            if saliency_method == 'prediction_diff':
                model_is_correct, saliency_scores, src_prn_id = \
                    _get_prediction_diff_scores(src, true_tgt, false_tgt, tgt_pron)
            else:
                model_is_correct = pd_saliency_table[qid][str(sid)]['model_is_correct']
                saliency_scores, src_prn_id = _get_gradient_norm_scores(src, true_tgt, tgt_pron)

            if model_is_correct is None:
                break

            # Update table
            saliency_entry['model_is_correct'] = model_is_correct
            saliency_entry['saliency_scores'] = saliency_scores
            saliency_entry['src_prn_id'] = src_prn_id
            # Merge scores into words (retain mean sub-word score per word)
            saliency_entry['word_saliency_scores'] = list()
            new_score_entry = ['@@', list()]
            for tpl in saliency_entry['saliency_scores']:
                if new_score_entry[0].endswith('@@'):
                    new_score_entry[0] = new_score_entry[0][:-2] + tpl[0]
                    # new_score_entry[1] = max(new_score_entry[1], tpl[1])
                    new_score_entry[1].append(tpl[1])
                else:
                    saliency_entry['word_saliency_scores'].append((new_score_entry[0], np.mean(new_score_entry[1])))
                    new_score_entry = [tpl[0], [tpl[1]]]
            saliency_entry['word_saliency_scores'].append(new_score_entry)

            # Check referent saliency
            # Handle multi-word referents
            saliency_ngram_set = list()
            ngram_lens = [len(ref.split()) for ref in [true_ref, false_ref] if len(ref.split()) > 1]
            for nl in ngram_lens:
                ngrams = list()
                for tpl_id, tpl in enumerate(saliency_entry['word_saliency_scores']):
                    if tpl_id < (len(saliency_entry['word_saliency_scores']) - nl + 1):
                        ngram = ' '.join([saliency_entry['word_saliency_scores'][tpl_id + n][0] for n in range(nl)])
                        score = np.mean([saliency_entry['word_saliency_scores'][tpl_id + n][1] for n in range(nl)])
                        ngrams.append([ngram, score])
                saliency_ngram_set.append(ngrams)

            try:
                saliency_entry['true_ref_score'] = [tpl[1] for tpl in saliency_entry['word_saliency_scores'] if
                                                    tpl[0].lower() == true_ref.lower()][-1]
            except IndexError:
                for ngrams in saliency_ngram_set:
                    try:
                        saliency_entry['true_ref_score'] = [tpl[1] for tpl in ngrams
                                                            if tpl[0].lower() == true_ref.lower()][-1]
                    except IndexError:
                        continue

            try:
                saliency_entry['false_ref_score'] = [tpl[1] for tpl in saliency_entry['word_saliency_scores'] if
                                                     tpl[0].lower() == false_ref.lower()][-1]
            except IndexError:
                for ngrams in saliency_ngram_set:
                    try:
                        saliency_entry['false_ref_score'] = [tpl[1] for tpl in ngrams
                                                             if tpl[0].lower() == false_ref.lower()][-1]
                    except IndexError:
                        continue

        # Check trigger scores
        saliency_table[qid][1]['trigger_saliency_scores'] = list()
        saliency_table[qid][2]['trigger_saliency_scores'] = list()
        # Check non-trigger scores (ignoring 'it')
        saliency_table[qid][1]['shared_saliency_scores'] = list()
        saliency_table[qid][2]['shared_saliency_scores'] = list()
        # Compute overlap and difference between contrastive English sentences
        alignment = pairwise2.align.globalxx([tpl[0] for tpl in saliency_table[qid][1]['saliency_scores']],
                                             [tpl[0] for tpl in saliency_table[qid][2]['saliency_scores']],
                                             gap_char=['<GAP>'])[0]
        seq_a_gaps, seq_b_gaps = 0, 0
        for tok_id, tok in enumerate(alignment.seqA):
            if tok == '<GAP>':
                seq_a_gaps += 1
            if alignment.seqB[tok_id] == '<GAP>':
                seq_b_gaps += 1
            if (tok_id - seq_a_gaps) != saliency_table[qid][1]['src_prn_id']:  # ignore ambiguous pronoun
                if tok != '<GAP>':  # ignore gaps
                    if tok == alignment.seqB[tok_id]:  # shared tokens
                        saliency_table[qid][1]['shared_saliency_scores'].append(
                            saliency_table[qid][1]['saliency_scores'][tok_id - seq_a_gaps])
                        saliency_table[qid][2]['shared_saliency_scores'].append(
                            saliency_table[qid][2]['saliency_scores'][tok_id - seq_b_gaps])
                    else:  # trigger tokens
                        saliency_table[qid][1]['trigger_saliency_scores'].append(
                            saliency_table[qid][1]['saliency_scores'][tok_id - seq_a_gaps])
                else:
                    if alignment.seqB[tok_id] != '<GAP>':
                        saliency_table[qid][2]['trigger_saliency_scores'].append(
                            saliency_table[qid][2]['saliency_scores'][tok_id - seq_b_gaps])

    # Dump saliency table to disk
    saliency_out_path = out_dir + '/saliency_table_{:s}.json'.format(saliency_method)
    print('Saving the saliency table to {:s}'.format(saliency_out_path))
    with open(saliency_out_path, 'w', encoding='utf8') as sop:
        json.dump(saliency_table, sop, indent=3, sort_keys=True, ensure_ascii=False)

    # Report
    # 1. Which referent is more salient when model is correct / incorrect)?
    true_ref_scores_model_correct = list()
    false_ref_scores_model_correct = list()
    true_ref_scores_model_incorrect = list()
    false_ref_scores_model_incorrect = list()
    for qid in saliency_table.keys():
        for sid in saliency_table[qid].keys():
            if saliency_table[qid][sid]['model_is_correct']:
                true_ref_scores_model_correct.append(saliency_table[qid][sid]['true_ref_score'])
                false_ref_scores_model_correct.append(saliency_table[qid][sid]['false_ref_score'])
            else:
                true_ref_scores_model_incorrect.append(saliency_table[qid][sid]['true_ref_score'])
                false_ref_scores_model_incorrect.append(saliency_table[qid][sid]['false_ref_score'])
    true_ref_scores_model_all = true_ref_scores_model_correct + true_ref_scores_model_incorrect
    false_ref_scores_model_all = false_ref_scores_model_correct + false_ref_scores_model_incorrect

    print('-' * 20)
    print('Resolved {:d} samples correctly'.format(len(true_ref_scores_model_correct)))
    print('Resolved {:d} samples incorrectly'.format(len(true_ref_scores_model_incorrect)))

    print('-' * 20)
    print('Mean (std.) saliency of the [CORRECT] referent in [CORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(true_ref_scores_model_correct), np.std(true_ref_scores_model_correct)))
    print('Mean (std.) saliency of the [INCORRECT] referent in [CORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(false_ref_scores_model_correct), np.std(false_ref_scores_model_correct)))
    print('Mean (std.) saliency of the [CORRECT] referent in [INCORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(true_ref_scores_model_incorrect), np.std(true_ref_scores_model_incorrect)))
    print('Mean (std.) saliency of the [INCORRECT] referent in [INCORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(false_ref_scores_model_incorrect), np.std(false_ref_scores_model_incorrect)))
    print('Mean (std.) saliency of the [CORRECT] referent in [ALL] samples: {:.4f} ({:.4f})'.format(
        np.mean(true_ref_scores_model_all), np.std(true_ref_scores_model_all)))
    print('Mean (std.) saliency of the [INCORRECT] referent in [ALL] samples: {:.4f} ({:.4f})'.format(
        np.mean(false_ref_scores_model_all), np.std(false_ref_scores_model_all)))

    print('TTest referents:')
    print(ttest(true_ref_scores_model_all, false_ref_scores_model_all))

    # 2. Are triggers more salient when the model correct?
    trigger_scores_model_correct = list()
    shared_scores_model_correct = list()
    trigger_scores_model_incorrect = list()
    shared_scores_model_incorrect = list()
    for qid in saliency_table.keys():
        for sid in saliency_table[qid].keys():
            if saliency_table[qid][sid]['model_is_correct']:
                trigger_scores_model_correct += [tpl[1] for tpl in saliency_table[qid][sid]['trigger_saliency_scores']]
                shared_scores_model_correct += [tpl[1] for tpl in saliency_table[qid][sid]['shared_saliency_scores']]
            else:
                trigger_scores_model_incorrect += [tpl[1]for tpl in saliency_table[qid][sid]['trigger_saliency_scores']]
                shared_scores_model_incorrect += [tpl[1] for tpl in saliency_table[qid][sid]['shared_saliency_scores']]
    trigger_scores_model_all = trigger_scores_model_correct + trigger_scores_model_incorrect
    shared_scores_model_all = shared_scores_model_correct + shared_scores_model_incorrect

    print('-' * 20)
    print('Average (std.) saliency of [TRIGGERS] in [CORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(trigger_scores_model_correct), np.std(trigger_scores_model_correct)))
    print('Average (std.) saliency of [SHARED TOKENS] in [CORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(shared_scores_model_correct), np.std(shared_scores_model_correct)))
    print('Average (std.) saliency of [TRIGGERS] in [INCORRECTLY scored] samples: {:.4f} ({:.4f})'.format(np.mean(
        trigger_scores_model_incorrect), np.std(trigger_scores_model_incorrect)))
    print('Average (std.) saliency of [SHARED TOKENS] in [INCORRECTLY scored] samples: {:.4f} ({:.4f})'.format(
        np.mean(shared_scores_model_incorrect), np.std(shared_scores_model_incorrect)))
    print('Average (std.) saliency of [TRIGGERS] in [ALL] samples: {:.4f} ({:.4f})'.format(
        np.mean(trigger_scores_model_all), np.std(trigger_scores_model_all)))
    print('Average (std.) saliency of [SHARED TOKENS] in [ALL] samples: {:.4f} ({:.4f})'.format(
        np.mean(shared_scores_model_all), np.std(shared_scores_model_all)))

    print('TTest triggers:')
    print(ttest(trigger_scores_model_all, shared_scores_model_all))

    # Make plots for 10 randomly drawn samples
    print('-' * 20)
    random.seed(42)
    qids = list(saliency_table.keys())
    random.shuffle(qids)
    plot_dir = '{:s}/plots_{:s}'.format(out_dir, saliency_method)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for qid in qids[:10]:
        for sid in saliency_table[qid].keys():
            plt_path = '{:s}/saliency_plot_{:s}-{:d}_{}.png'.format(
                plot_dir, qid, sid, saliency_table[qid][sid]['model_is_correct'])
            plot_saliency_map(saliency_table[qid][sid]['saliency_scores'], '@@', plt_path)
    print('Saved all plots to {:s}'.format(out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='path to the JSON file containing the contrastive samples')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='path to the directory containing checkpoints')
    parser.add_argument('--checkpoint_names', type=str, required=True,
                        help='name(s) of checkpoints to use')
    parser.add_argument('--use_cpu',  action='store_true',
                        help='whether to use the CPU for model passes'),
    parser.add_argument('--use_subword_nmt', action='store_true',
                        help='set to TRUE when evaluating the WMT14 en-fr model')
    parser.add_argument('--src_codes_path', type=str, required=True,
                        help='path to the files containing the source BPE-codes of the evaluated model')
    parser.add_argument('--tgt_codes_path', type=str, required=True,
                        help='path to the files containing the target BPE-codes of the evaluated model')
    parser.add_argument('--src_vocab_path', type=str, default=None,
                        help='path to the file containing model source vocabulary')
    parser.add_argument('--tgt_vocab_path', type=str, default=None,
                        help='path to the file containing model target vocabulary')
    parser.add_argument('--src_lang', type=str,
                        help='language code corresponding to the source language')
    parser.add_argument('--tgt_lang', type=str,
                        help='language code corresponding to the target language')
    parser.add_argument('--saliency_method', type=str, choices=['prediction_diff', 'gradient_norm'],
                        help='saliency method to be used for the performed analysis; NOTE: Due to the construction '
                             'of the evaluation protocol, \'gradient_norm\' evaluation can only be run after '
                             '\'prediction_diff\' has been completed once')
    parser.add_argument('--pd_saliency_table_path', type=str, default=None,
                        help='path to the JSON file containing the results of the \'prediction_diff\' evaluation')
    args = parser.parse_args()

    # Create output directory, if necessary
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Load translation model
    if args.saliency_method == 'gradient_norm':
        assert args.pd_saliency_table_path is not None, \
            '\'prediction_diff\' evaluation must be completed before running \'gradient_norm\' evaluation'

    # Load model
    checkpoints_to_load = \
        [args.checkpoint_dir + '{:s}'.format(name) for name in args.checkpoint_names.split(':')]
    arg_overrides = {'cpu': args.use_cpu}
    models, saved_cfg, task = \
        checkpoint_utils.load_model_ensemble_and_task(checkpoints_to_load, arg_overrides=arg_overrides)
    model = models[0]

    device = 'cpu' if args.use_cpu else 'cuda:0'

    model.eval()
    if device != 'cpu':
        model.cuda()

    # Build criterion
    if args.saliency_method == 'gradient_norm':
        criterion = task.build_criterion('label_smoothed_cross_entropy_at_position')
    else:
        criterion = task.build_criterion('label_smoothed_cross_entropy_with_probs')
    criterion.eval()

    # Zero-out an unused embedding in the translation model
    with torch.no_grad():
        mask_index = model.encoder.dictionary.index('madeupword0000')
        model.encoder.embed_tokens.weight[mask_index, :] *= 0  # zero-out embedding

    # Initialize tokenizers
    src_mt = MosesTokenizer(args.src_lang)
    tgt_mt = MosesTokenizer(args.tgt_lang)

    # Initialize BPE
    if args.use_subword_nmt:
        with open(args.src_vocab_path, 'r', encoding='utf8') as sv:
            src_vocab = read_vocabulary(sv, 50)
        with open(args.tgt_vocab_path, 'r', encoding='utf8') as tv:
            tgt_vocab = read_vocabulary(tv, 50)
        with open(args.src_codes_path, 'r', encoding='utf8') as sc:
            src_bpe_model = BPE(sc, vocab=src_vocab)
        with open(args.tgt_codes_path, 'r', encoding='utf8') as tc:
            tgt_bpe_model = BPE(tc, vocab=tgt_vocab)
    else:
        src_vocab, tgt_vocab = None, None
        src_bpe_model = fastBPE.fastBPE(args.src_codes_path, args.src_vocab_path)
        tgt_bpe_model = fastBPE.fastBPE(args.tgt_codes_path, args.tgt_vocab_path)

    compute_saliency_scores(args.json_file_path, args.out_dir, args.saliency_method, args.pd_saliency_table_path)
