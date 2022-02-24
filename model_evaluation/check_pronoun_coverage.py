import math
import stanza
import argparse
import numpy as np

from scipy.stats import chisquare


def check_iprovements(ref_tgt_path, base_tgt_path, ft_tgt_path,
                      ref_alignment_path, base_alignment_path, ft_alignment_path):
    """ Checks whether fine-tuning improved pronoun translation quality. """

    def _jaccard_similarity(l1, l2):
        """ Computes the Jaccard Similarity between pronoun translations in both lists. """
        d1 = {p: l1.count(p) for p in l1}
        d2 = {p: l2.count(p) for p in l2}
        intersection = 0
        for k in d1.keys():
            intersection += min(d1.get(k, 0), d2.get(k, 0))
        return intersection / (len(l1) + len(l2) - intersection)

    def _get_align_table(align_line):
        """ Transforms an alignment line into a look-up table. """
        align_table = dict()
        for st in align_line.split():
            s, t = st.split('-')
            s = int(s)
            t = int(t)
            if s not in align_table.keys():
                align_table[s] = list()
            align_table[s].append(t)
        return align_table

    def _cohens_w(observed_proportions, expected_proportions):
        """ Computes Cohen's W """
        parts = list()
        for p_id, op in enumerate(observed_proportions):
            parts.append(((op - expected_proportions[p_id]) ** 2) / expected_proportions[p_id])
        return math.sqrt(sum(parts))

    def _get_stats(gender_list, ref_table, model_table):
        """ Performs statistical analysis on the pronoun distributions """
        ref_counts = [ref_table[g] for g in gender_list]
        sum_ref_counts = sum(ref_counts)
        ref_proportions = [c / sum_ref_counts for c in ref_counts]
        model_counts = [model_table[g] for g in gender_list]
        sum_model_counts = sum(model_counts)
        model_proportions = [c / sum_model_counts for c in model_counts]

        p = chisquare(f_obs=model_counts, f_exp=ref_counts)
        w = _cohens_w(model_proportions, ref_proportions)
        return p, w

    # Read
    with open(ref_tgt_path, 'r', encoding='utf8') as ref_tgt_f:
        ref_tgt_lines = ref_tgt_f.readlines()
    with open(base_tgt_path, 'r', encoding='utf8') as base_tgt_f:
        base_tgt_lines = base_tgt_f.readlines()
    with open(ft_tgt_path, 'r', encoding='utf8') as tf_tgt_f:
        ft_tgt_lines = tf_tgt_f.readlines()
    with open(ref_alignment_path, 'r', encoding='utf8') as ref_alg_f:
        ref_alignments = ref_alg_f.readlines()
    with open(base_alignment_path, 'r', encoding='utf8') as base_alg_f:
        base_alignments = base_alg_f.readlines()
    with open(ft_alignment_path, 'r', encoding='utf8') as ft_alg_f:
        ft_alignments = ft_alg_f.readlines()

    # Check pronoun translations
    num_it_mentions = 0
    ref_prn_trans = {'Masc': 0, 'Fem': 0, 'Neut': 0}
    ref_it_trans = {'Masc': 0, 'Fem': 0, 'Neut': 0}

    base_prn_trans = {'Masc': 0, 'Fem': 0, 'Neut': 0}
    base_it_trans = {'Masc': 0, 'Fem': 0, 'Neut': 0}
    base_sent_js = list()   # mean Jaccard overlap between pronouns in refrence and model translation (higher == better)
    base_it_errors = 0

    ft_prn_trans = {'Masc': 0, 'Fem': 0, 'Neut': 0}
    ft_it_trans = {'Masc': 0, 'Fem': 0, 'Neut': 0}
    ft_sent_js = list()
    ft_it_errors = 0

    genders = list(ref_prn_trans.keys())

    for line_id, line in enumerate(ref_tgt_lines):

        if line_id > 0 and (line_id + 1) % 100 == 0:
            try:
                print('Processing line {:d}'.format(line_id))
                print(num_it_mentions)
                print(ref_prn_trans)
                print(ref_it_trans)
                print('-')
                print(base_prn_trans)
                print(base_it_trans)
                print(np.mean(base_sent_js), base_it_errors)
                print('-')
                print(ft_prn_trans)
                print(ft_it_trans)
                print(np.mean(ft_sent_js), ft_it_errors)
                print('-')
                print(_get_stats(genders, ref_prn_trans, base_prn_trans))
                print(_get_stats(genders, ref_it_trans, base_it_trans))
                print('-')
                print(_get_stats(genders, ref_prn_trans, ft_prn_trans))
                print(_get_stats(genders, ref_it_trans, ft_it_trans))
                print('=' * 10)
            except Exception:
                continue

        src, ref_tgt = line.split(' ||| ')
        src_parse = nlp_src(src)
        ref_tgt_parse = nlp_tgt(ref_tgt)
        ref_align = ref_alignments[line_id]
        ref_align_dict = _get_align_table(ref_align)

        _, base_tgt = base_tgt_lines[line_id].split(' ||| ')
        base_tgt_parse = nlp_tgt(base_tgt)
        base_align = base_alignments[line_id]
        base_align_dict = _get_align_table(base_align)

        _, ft_tgt = ft_tgt_lines[line_id].split(' ||| ')
        ft_tgt_parse = nlp_tgt(ft_tgt)
        ft_align = ft_alignments[line_id]
        ft_align_dict = _get_align_table(ft_align)

        # Check target pronouns
        ref_prn_list, base_prn_list, ft_prn_list = list(), list(), list()
        for parse, prn_trans, prn_list in [(ref_tgt_parse, ref_prn_trans, ref_prn_list),
                                           (base_tgt_parse, base_prn_trans, base_prn_list),
                                           (ft_tgt_parse, ft_prn_trans, ft_prn_list)]:
            for s in parse.sentences[:1]:
                for w in s.words:
                    if w.upos == 'PRON':
                        prn_list.append(w.text.lower())
                        # Check gender
                        try:
                            feats = {f.split('=')[0]: f.split('=')[1] for f in w.feats.split('|')}
                            gen = feats.get('Gender', None)
                            if gen is not None:
                                prn_trans[gen] += 1
                        except AttributeError:
                            continue

        # Check pronoun overlap
        if len(ref_prn_list) > 0:
            base_sent_js.append(_jaccard_similarity(ref_prn_list, base_prn_list))
            ft_sent_js.append(_jaccard_similarity(ref_prn_list, ft_prn_list))

        # Check 'it'-translations (assumes one sentence per line)
        for sent in src_parse.sentences[:1]:
            for w_id, src_w in enumerate(sent.words):
                if src_w.text.lower() == 'it':
                    num_it_mentions += 1
                    ref_gen, base_gen, ft_gen = None, None, None
                    # Check reference translation
                    ref_align_loc = ref_align_dict.get(w_id, [])
                    for loc in ref_align_loc:
                        if loc >= len(ref_tgt_parse.sentences[0].words):
                            continue
                        ref_w = ref_tgt_parse.sentences[0].words[loc]
                        if ref_w.upos == 'PRON':
                            # Check gender
                            try:
                                feats = {f.split('=')[0]: f.split('=')[1] for f in ref_w.feats.split('|')}
                                gen = feats.get('Gender', None)
                                if gen is not None:
                                    ref_gen = gen
                                    ref_it_trans[ref_gen] += 1
                            except AttributeError:
                                continue

                    # Check BASE translation
                    base_align_loc = base_align_dict.get(w_id, [])
                    for loc in base_align_loc:
                        if loc >= len(base_tgt_parse.sentences[0].words):
                            continue
                        base_w = base_tgt_parse.sentences[0].words[loc]
                        if base_w.upos == 'PRON':
                            # Check gender
                            try:
                                feats = {f.split('=')[0]: f.split('=')[1] for f in base_w.feats.split('|')}
                                gen = feats.get('Gender', None)
                                if gen is not None:
                                    base_gen = gen
                                    base_it_trans[base_gen] += 1
                            except AttributeError:
                                continue

                    # Check FT translation
                    ft_align_loc = ft_align_dict.get(w_id, [])
                    for loc in ft_align_loc:
                        if loc >= len(ft_tgt_parse.sentences[0].words):
                            continue
                        ft_w = ft_tgt_parse.sentences[0].words[loc]
                        if ft_w.upos == 'PRON':
                            # Check gender
                            try:
                                feats = {f.split('=')[0]: f.split('=')[1] for f in ft_w.feats.split('|')}
                                gen = feats.get('Gender', None)
                                if gen is not None:
                                    ft_gen = gen
                                    ft_it_trans[ft_gen] += 1
                            except AttributeError:
                                continue

                    # Update errors
                    if ref_gen != base_gen:
                        base_it_errors += 1
                    if ref_gen != ft_gen:
                        ft_it_errors += 1

    # Report
    print('=' * 20)
    print('Found {} mentions of "it" in the source sentence'.format(num_it_mentions))
    print('Reference pronoun translations: {}'.format(ref_prn_trans))
    print('Reference "it" translations: {}'.format(ref_it_trans))
    print('-' * 10)
    print('Base pronoun translations: {}'.format(base_prn_trans))
    print('Base "it" translations: {}'.format(base_it_trans))
    print('Base mean JS: {}'.format(np.mean(base_sent_js)))
    print('Base "it" errors: {}'.format(base_it_errors))
    print('-' * 10)
    print('FT pronoun translations: {}'.format(ft_prn_trans))
    print('FT "it" translations: {}'.format(ft_it_trans))
    print('FT mean JS: {}'.format(np.mean(ft_sent_js)))
    print('FT "it" errors: {}'.format(ft_it_errors))

    # Do some significance testing
    prn_base_p, prn_base_w = _get_stats(genders, ref_prn_trans, base_prn_trans)
    it_base_p, it_base_w = _get_stats(genders, ref_it_trans, base_it_trans)
    prn_ft_p, prn_ft_w = _get_stats(genders, ref_prn_trans, ft_prn_trans)
    it_ft_p, it_ft_w = _get_stats(genders, ref_it_trans, ft_it_trans)
    print('=' * 10)
    print('Base model stats for pronoun translation: p = {}, w = {}'.format(prn_base_p, prn_base_w))
    print('Base model stats for "it"" translation: p = {}, w = {}'.format(it_base_p, it_base_w))
    print('-' * 10)
    print('FT model stats for pronoun translation: p = {}, w = {}'.format(prn_ft_p, prn_ft_w))
    print('FT model stats for "it"" translation: p = {}, w = {}'.format(it_ft_p, it_ft_w))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # All translations are in the "src ||| tgt" format required by aligners and are Moses-tokenized
    parser.add_argument('--ref_tgt_path', type=str, required=True,
                        help='path to reference translations')
    parser.add_argument('--base_tgt_path', type=str, required=True,
                        help='path to translations by the baseline model')
    parser.add_argument('--ft_tgt_path', type=str, required=True,
                        help='path to translations by the fine-tuned model')
    parser.add_argument('--ref_alignment_path', type=str, required=True,
                        help='path to alignments with reference translation')
    parser.add_argument('--base_alignment_path', type=str, required=True,
                        help='path to alignments with reference translation')
    parser.add_argument('--ft_alignment_path', type=str, required=True,
                        help='path to alignments with reference translation')
    parser.add_argument('--tgt_lang', type=str, required=True,
                        help='target language ID')
    args = parser.parse_args()

    # Initialize Stanza parser for the target language
    nlp_src = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True)
    nlp_tgt = stanza.Pipeline(args.tgt_lang, processors='tokenize,pos', use_gpu=True)

    check_iprovements(args.ref_tgt_path, args.base_tgt_path, args.ft_tgt_path,
                      args.ref_alignment_path, args.base_alignment_path, args.ft_alignment_path)
