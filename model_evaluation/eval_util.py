""" Various evaluation helper scripts. """

import json
import torch
import string
import random
import logging

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def read_jsonl(input_file):
    """ Helper for reading .jsonl files """
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_sublist_indices(l, sl):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))
    return results


def get_final_index(seq_tokens, item):
    """ Returns the final index of the item within the provided sequence. """
    # Split and reverse sequence
    if type(seq_tokens[0]) == str:
        seq_tokens = [t.lower().strip() for t in seq_tokens]
        # For multi-token items, only consider the final token
        item = item.split()[-1].lower()

    # Identify item index
    rev_seq_tokens = seq_tokens[:]
    rev_seq_tokens.reverse()
    assert item in rev_seq_tokens, 'Could not find item {} in sequence {}'.format(item, rev_seq_tokens)
    return len(rev_seq_tokens) - 1 - rev_seq_tokens.index(item)


def build_sample(src_tokens, task, device):
    """ Creates a fairseq sample """

    from fairseq import utils

    dataset = task.build_dataset_for_inference(src_tokens, [x.numel() for x in src_tokens])
    sample = dataset.collater(dataset)
    sample = utils.apply_to_sample(lambda tensor: tensor.to(device), sample)
    return sample


def segment_seq(seq, tokenizer, bpe_model, model_dict, use_subword_nmt):
    """ Helper function used segment model inputs """
    tokenized_seq = tokenizer.tokenize(seq)
    if use_subword_nmt:
        seq_bpe = bpe_model.process_line(' '.join(tokenized_seq)).split()
    else:
        seq_bpe = bpe_model.apply(tokenized_seq)
        seq_bpe = ' '.join(seq_bpe).split()  # splits words containing sub-words
    seq_ids = torch.tensor([model_dict.index(bpe_tok) for bpe_tok in seq_bpe] + [model_dict.eos_index])
    return seq_bpe, seq_ids


def prepare_inputs(src_ids, tgt_ids, model, task, device):
    """ Helper function used to prepare model inputs """

    from fairseq.data.data_utils import collate_tokens

    sample = build_sample([src_ids], task, device)
    eos_idx, pad_idx = model.decoder.dictionary.eos_index, model.decoder.dictionary.pad_index
    sample['net_input']['prev_output_tokens'] = \
        collate_tokens([tgt_ids], pad_idx, eos_idx, move_eos_to_beginning=True).to(device)
    sample['target'] = collate_tokens([tgt_ids], pad_idx, eos_idx).to(device)
    return sample


def compute_perplexity(token_probabilities):
    """ Computes perplexity on the basis of token-level probabilities """
    # Compute PPL from probabilities
    return np.prod([1 / p for p in token_probabilities]) ** (1. / len(token_probabilities))


def count_distance_in_subwords(sequence, true_referent, target, codes, bpe_model, vocab, tokenizer):
    """ Counts the distance (in sub-words) between the true referent and the ambiguous pronoun on the SOURCE side. """

    from subword_nmt.apply_bpe import check_vocab_and_split

    # Tokenize
    seq_tokens = sequence.split()
    if target == '_':
        seq_tokens = [t.strip(string.punctuation.replace('_', '')) for t in seq_tokens]
    # Isolate sub-sequence between referent and pronoun
    ref_loc = get_final_index(seq_tokens, true_referent)
    pron_loc = get_final_index(seq_tokens, target)
    sub_seq = ' '.join(seq_tokens[ref_loc + 1: pron_loc])
    # Segment the sub-sequence and return its length
    tokenizer_out = tokenizer.tokenize(sub_seq)
    if bpe_model is None and vocab is None:
        return len(tokenizer_out)
    if len(tokenizer_out) == 0:
        return 0
    if bpe_model is None:
        # Use subword-nmt
        return len(check_vocab_and_split(' '.join(tokenizer_out), codes, vocab, '@@'))
    else:
        # Use fastBPE
        return len(bpe_model.apply([' '.join(tokenizer_out)])[0])


def make_bar_plot(entries_dict, x_label, y_label, groups, out_dir, file_name):
    """ Creates a comparative bar plot. """

    def _label_heights(category):
        """ Attach a text label above each bar displaying its height """
        for c in category:
            height = c.get_height()
            plt.text(c.get_x() + c.get_width() / 2., height, '%d' % int(height),
                     ha='center', va='bottom', size='x-small')

    # Prepare
    bars1, bars2, bars3, bars4 = list(), list(), list(), list()
    for g in groups:
        bars1.append(entries_dict['reference_true'][g])
        bars2.append(entries_dict['model_preference'][g])
        bars3.append(entries_dict['reference_false'][g])
        bars4.append(entries_dict['model_rejection'][g])

    # Set bar positions
    width = 0.20
    r1 = np.arange(len(groups))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]

    # Plot
    with plt.style.context('seaborn-paper'):
        # plt.figure(figsize=(8, 6))

        cat1 = plt.bar(
            r1, bars1, width=width, edgecolor='black', label='reference_true', color='#67a9cf')  # hatch='/'
        cat2 = plt.bar(
            r2, bars2, width=width, edgecolor='black', label='model_preference', color='#2166ac')  # hatch='+'
        cat3 = plt.bar(
            r3, bars3, width=width, edgecolor='black', label='reference_false', color='#ef8a62')  # hatch='//'
        cat4 = plt.bar(
            r4, bars4, width=width, edgecolor='black', label='model_rejection', color='#b2182b')  # hatch='x'

        for cat in [cat1, cat2, cat3, cat4]:
            _label_heights(cat)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks([r + (1.5 * width) for r in range(len(bars1))], groups)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        fig_path = '{:s}/{:s}'.format(out_dir, file_name)
        plt.savefig(fig_path)
        print('=== Saved bar plot to {:s} ==='.format(fig_path))
        plt.close()


def make_density_plot(entries_dict, columns, out_dir, file_name):
    """ Plots the densities corresponding to references / model preferences for easy comparison """
    # Convert to dataframe
    col_a = list()
    col_b = list()
    for key in entries_dict.keys():
        if type(entries_dict[key]) == list:
            sample_size = min([len(entries_dict[key]) for key in entries_dict.keys()])
            col_a += [key] * sample_size
            values = entries_dict[key]
            random.shuffle(values)
            col_b += values[:sample_size]
        else:
            for val in entries_dict[key].keys():
                col_a += [key] * entries_dict[key][val]
                col_b += [val] * entries_dict[key][val]

    df = pd.DataFrame(list(zip(col_a, col_b)), columns=columns)

    # Plot
    with plt.style.context('seaborn-paper'):
        sns.kdeplot(x=df[columns[1]], hue=df[columns[0]], fill=True, multiple='layer')

        fig_path = '{:s}/{:s}'.format(out_dir, file_name)
        plt.savefig(fig_path)
        print('=== Saved density plot to {:s} ==='.format(fig_path))
        plt.close()



def plot_saliency_map(saliency_scores, separator, plot_path):
    """ Creates a saliency plot that highlights salient tokens """
    # see https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
    # Normalize scores
    tokens, scores = zip(*saliency_scores)
    # Reduce the score disparity by taking the n-th root for more readable plots
    scores = [(abs(s) ** (1 / 1.5)) if s > 0 else (-1 * (abs(s) ** (1 / 1.5))) for s in scores]
    num = max(scores) * 1.5  # avoid darkest colors
    scores = [s / num for s in scores]

    # Prepare plot
    cmap = matplotlib.cm.get_cmap('Reds')  # colormap
    start_x = 20  # coordinates of the initial word
    start_y = 450
    end = 600
    figure = plt.figure()
    # Assign background colors to words
    rend = figure.canvas.get_renderer()

    for tok, score in zip(tokens, scores):
        color = matplotlib.colors.rgb2hex(cmap(score)[:3])  # obtain color value
        tok_box = dict(fc='{:s}'.format(color), linewidth=0.0, snap=True, pad=2.0)
        txt = plt.text(start_x, start_y, tok, color='black', bbox=tok_box, transform=None)  # plot word
        box_width = txt.get_window_extent(renderer=rend)

        if separator == '@@':
            whitespace = 5 if tok.endswith(separator) else 10
        else:
            # assumes separator is ▁
            whitespace = 5 if not tok.endswith(separator) else 10

        start_x = box_width.width + start_x + whitespace  # placement of the subsequent word
        if start_x >= end:
            start_x = 20
            start_y -= 40

    # Skip plotting axis
    plt.axis('off')
    # Save figure
    plt.savefig(plot_path)

