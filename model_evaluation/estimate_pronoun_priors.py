import math
import string
import argparse

from scipy.stats import chisquare


LANG2PRON = {'de': {'er': 'male', 'sie': 'female', 'es': 'neutral'},
             'fr': {'il': 'male', 'elle': 'female', 'ils': 'male', 'elles': 'female'},
             'ru': {'он': 'male', 'она': 'female', 'оно': 'neutral'}}


def compute_priors(src_file_path, tgt_file_path, alignments_path, tgt_lang):
    """ Counts the occurrence of pronoun base forms in the (Moses-tokenized) training data """

    # Initialize trackers
    absolute_counts = {'male': 0, 'female': 0, 'neutral': 0}
    aligned_counts = {'male': 0, 'female': 0, 'neutral': 0}

    pronouns = LANG2PRON[tgt_lang]

    # Look through the files
    alignments = None
    pron_src = list()
    pron_tgt = list()
    if alignments_path is not None:
        with open(alignments_path, 'r', encoding='utf8') as aln_file:
            alignments = aln_file.readlines()
    with open(src_file_path, 'r', encoding='utf8') as src_file:
        src_lines = src_file.readlines()

    with open(tgt_file_path, 'r', encoding='utf8') as tgt_file:
        for line_id, line in enumerate(tgt_file):

            if line_id > 0 and (line_id + 1) % 10000 == 0:
                print('-' * 10)
                print('Evaluated {:d} lines'.format(line_id + 1))
                print('Absolute counts:')
                print(absolute_counts)
                print('Aligned counts:')
                print(aligned_counts)

            # De-bpe
            line = line.replace('@@ ', '')
            src_line = src_lines[line_id].replace('@@ ', '')

            # Update absolute counts
            has_pron = False
            tgt_tokens = line.lower().strip().split()
            for tok in tgt_tokens:
                if pronouns.get(tok.lower(), None) is not None:
                    absolute_counts[pronouns[tok]] += 1
                    has_pron = True

            if alignments is None:
                if has_pron:
                    # Store pairs with pronouns for alignment computation
                    pron_src.append(src_line)
                    pron_tgt.append(line)
            else:
                # Update aligned counts
                src_tokens = src_line.lower().strip().split()
                line_align = dict()
                for tok_id, tok in enumerate(src_tokens):
                    if tok.strip(string.punctuation) == 'it':
                        if len(line_align) == 0:
                            for pair in alignments[line_id].split():
                                src_l, tgt_l = pair.split('-')
                                src_l = int(src_l)
                                tgt_l = int(tgt_l)
                                if line_align.get(src_l, None) is None:
                                    line_align[src_l] = [tgt_l]
                                else:
                                    line_align[src_l].append(tgt_l)
                        it_trans = line_align.get(tok_id, [])
                        if len(it_trans) > 0:
                            for tl in it_trans:
                                if pronouns.get(tgt_tokens[tl], None) is not None:
                                    aligned_counts[pronouns[tgt_tokens[tl]]] += 1

    print('=' * 20)
    print('Finished!')
    print('Absolute counts:')
    print(absolute_counts)
    print('-' * 10)
    print('Aligned counts:')
    print(aligned_counts)

    if len(pron_src) > 0:
        print('Found {:d} pairs with pronouns'.format(len(pron_src)))
        pron_src_path = '{:s}.{:s}'.format(src_file_path, 'with_pronouns')
        with open(pron_src_path, 'w', encoding='utf8') as sp_f:
            for line in pron_src:
                sp_f.write(line)
        pron_tgt_path = '{:s}.{:s}'.format(tgt_file_path, 'with_pronouns')
        with open(pron_tgt_path, 'w', encoding='utf8') as tp_f:
            for line in pron_tgt:
                tp_f.write(line)

    # Perform Chi-Square Goodness of Fit test (with uniform being the zero-hypothesis distribution) and estimate
    #  effect size as Cohen's W (https://www.spss-tutorials.com/chi-square-goodness-of-fit-test/#effect-size-cohens-w)
    absolute_count_values = [absolute_counts['male'], absolute_counts['female'], absolute_counts['neutral']]
    aligned_count_values = [aligned_counts['male'], aligned_counts['female'], aligned_counts['neutral']]
    if tgt_lang == 'fr':
        absolute_count_values = absolute_count_values[:-1]
        aligned_count_values = aligned_count_values[:-1]
        del absolute_counts['neutral']
        del aligned_counts['neutral']
    sum_absolute_counts = sum(absolute_count_values)
    sum_aligned_counts = sum(aligned_count_values)

    absolute_uniform_counts = [sum_absolute_counts // len(absolute_count_values) for _ in absolute_count_values]
    absolute_uniform_proportion = 1 / len(absolute_count_values)
    absolute_observed_proportions = [c / sum_absolute_counts for c in absolute_count_values]

    aligned_uniform_counts = [sum_aligned_counts // len(aligned_count_values) for _ in aligned_count_values]
    aligned_uniform_proportion = 1 / len(aligned_count_values)
    aligned_observed_proportions = [c / sum_aligned_counts for c in aligned_count_values]

    absolute_p = chisquare(f_obs=absolute_count_values, f_exp=absolute_uniform_counts)
    aligned_p = chisquare(f_obs=aligned_count_values, f_exp=aligned_uniform_counts)

    def _cohens_w(observed_proportions, uniform_proportion):
        """ Computes Cohen's W """
        parts = list()
        for op in observed_proportions:
            parts.append(((op - uniform_proportion) ** 2) / uniform_proportion)
        return math.sqrt(sum(parts))

    absolute_w = _cohens_w(absolute_observed_proportions, absolute_uniform_proportion)
    aligned_w = _cohens_w(aligned_observed_proportions, aligned_uniform_proportion)

    print('-' * 20)
    print('Absolute Chi-Square results:')
    print('p : ', absolute_p)
    print('w : ', absolute_w)
    print('-' * 5)
    print('Aligned Chi-Square results:')
    print('p : ', aligned_p)
    print('w : ', aligned_w)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file_path', type=str, required=True,
                        help='path to the source side of a bi-text')
    parser.add_argument('--tgt_file_path', type=str, required=True,
                        help='path to the target side of a bi-text')
    parser.add_argument('--alignments_path', type=str, default=None,
                        help='path to the alignments file')
    parser.add_argument('--tgt_lang', type=str, required=True,
                        help='target language ID')

    args = parser.parse_args()

    compute_priors(args.src_file_path, args.tgt_file_path, args.alignments_path, args.tgt_lang)
