import string
import fastBPE
import argparse

import numpy as np

from pingouin import mwu
from sacremoses import MosesTokenizer
from subword_nmt.apply_bpe import read_vocabulary

from eval_util import \
    read_jsonl, get_final_index, count_distance_in_subwords, count_attractors, make_bar_plot, make_density_plot


LANG2PRON = {'de': {'er': 'male', 'sie': 'female', 'es': 'neutral'},
             'fr': {'il': 'male', 'elle': 'female', 'ils': 'male', 'elles': 'female'},
             'ru': {'oн': 'male', 'oнa': 'female', 'oнo': 'neutral'}}


def compute_accuracy(samples_path, scores_path, labels_path, codes_path, tgt_lang):
    """ Computes model accuracy given a list of model prediction scores and reference labels;
    compatible with all evaluated tasks """

    # Read-in
    samples = read_jsonl(samples_path)
    scores = list()
    labels = list()

    for array, path in [(scores, scores_path),
                        (labels, labels_path)]:
        with open(path, 'r', encoding='utf8') as f:
            pair = list()
            for line in f:
                entry = float(line.strip())
                pair.append(entry)
                if len(pair) == 2:
                    array.append(pair)
                    pair = list()

    # Compute accuracy and track input features that may be indicative of the model using heuristics
    num_correct, num_incorrect, num_correct_pairs, num_incorrect_pairs = 0, 0, 0, 0

    gender_counts = {'reference_true': {v: 0 for v in PRON2GEN.values()},
                     'reference_false': {v: 0 for v in PRON2GEN.values()},
                     'model_preference': {v: 0 for v in PRON2GEN.values()},
                     'model_rejection': {v: 0 for v in PRON2GEN.values()}}

    ref_gender_counts_per_position = {'first': {v: 0 for v in PRON2GEN.values()},
                                      'second': {v: 0 for v in PRON2GEN.values()}}  # for correct translation only

    position_counts = {'reference_true': {'first': 0, 'second': 0},
                       'reference_false': {'first': 0, 'second': 0},
                       'model_preference': {'first': 0, 'second': 0},
                       'model_rejection': {'first': 0, 'second': 0}}

    token_distance = \
        {'reference_true': dict(), 'reference_false': dict(), 'model_preference': dict(), 'model_rejection': dict()}
    # This is intended to estimate the "influence" of the trigger phrases
    perplexity_changes_trigger = {'same_choice': list(), 'both_right': list(), 'both_wrong': list()}
    # This is intended to estimate the "influence" of the target pronoun
    perplexity_changes_pronoun = {'correct': list(), 'incorrect': list()}

    assert len(samples) == len(scores) == len(labels), \
        'Unequal number of dataset samples, model scores, and sample labels'

    contrastive_pair_ppl = list()
    contrastive_pair_successes = list()
    for sample_id in range(len(scores)):

        # Report
        if (sample_id - 1) > 0 and (sample_id - 1) % 100 == 0:
            print('Processed {:d} samples'.format(sample_id - 1))

        score_pair = scores[sample_id]
        label_pair = labels[sample_id]

        # Identify filler gender in both translations
        try:
            gen_pair = [PRON2GEN[samples[sample_id]['pronoun1']], PRON2GEN[samples[sample_id]['pronoun2']]]
        except KeyError:
            gen_pair = [None, None]

        true_ref_gen = gen_pair[label_pair.index(max(label_pair))]
        false_ref_gen = gen_pair[label_pair.index(min(label_pair))]

        # Establish the relative position of the correct referent
        sentence_tokens = [tok.strip(string.punctuation) for tok in samples[sample_id]['sentence'].split()]
        if int(samples[sample_id]['answer']) == 1:
            true_referent_is_first = get_final_index(sentence_tokens, samples[sample_id]['referent1_en']) < \
                                     get_final_index(sentence_tokens, samples[sample_id]['referent2_en'])
        else:
            true_referent_is_first = get_final_index(sentence_tokens, samples[sample_id]['referent2_en']) < \
                                     get_final_index(sentence_tokens, samples[sample_id]['referent1_en'])
        if true_referent_is_first:
            position_counts['reference_true']['first'] += 1
            position_counts['reference_false']['second'] += 1
            if true_ref_gen is not None:
                ref_gender_counts_per_position['first'][true_ref_gen] += 1
                ref_gender_counts_per_position['second'][false_ref_gen] += 1
        else:
            position_counts['reference_true']['second'] += 1
            position_counts['reference_false']['first'] += 1
            if true_ref_gen is not None:
                ref_gender_counts_per_position['first'][false_ref_gen] += 1
                ref_gender_counts_per_position['second'][true_ref_gen] += 1

        # Measure sub-word distance
        distance_pair = [count_distance_in_subwords(samples[sample_id]['sentence'], samples[sample_id]['referent1_en'],
                                                    'it', codes_path, bpe_model, vocab, src_mt),
                         count_distance_in_subwords(samples[sample_id]['sentence'], samples[sample_id]['referent2_en'],
                                                    'it', codes_path, bpe_model, vocab, src_mt)]

        # Check model preferences
        if score_pair[0] == score_pair[1]:
            num_incorrect += 1
            contrastive_pair_successes.append(0)
        else:
            min_ppl_id = score_pair.index(min(score_pair))
            correct_id = label_pair.index(max(label_pair))

            # Track model accuracy
            if min_ppl_id == correct_id:
                num_correct += 1
                contrastive_pair_successes.append(1)
            else:
                num_incorrect += 1
                contrastive_pair_successes.append(0)

            # Track model perplexity for the preferred translation
            contrastive_pair_ppl.append((min_ppl_id == correct_id, score_pair[min_ppl_id]))

            # Track heuristics
            # Gender
            if true_ref_gen is not None:
                gender_counts['reference_true'][gen_pair[correct_id]] += 1
                gender_counts['reference_false'][gen_pair[abs(1 - correct_id)]] += 1
                gender_counts['model_preference'][gen_pair[min_ppl_id]] += 1
                gender_counts['model_rejection'][gen_pair[abs(1 - min_ppl_id)]] += 1

            # Relative referent position
            if true_referent_is_first:
                pref_key, rej_key = ('first', 'second') if min_ppl_id == correct_id else ('second', 'first')
            else:
                pref_key, rej_key = ('second', 'first') if min_ppl_id == correct_id else ('first', 'second')
            position_counts['model_preference'][pref_key] += 1
            position_counts['model_rejection'][rej_key] += 1

            # Sub-word distance
            for key, tpl_id in [('reference_true', correct_id), ('reference_false', abs(1 - correct_id)),
                                ('model_preference', min_ppl_id), ('model_rejection', abs(1 - min_ppl_id))]:
                if token_distance[key].get(distance_pair[tpl_id], None) is None:
                    token_distance[key][distance_pair[tpl_id]] = 0
                token_distance[key][distance_pair[tpl_id]] += 1

            # Track perplexity changes within the sample
            ppl_diff = abs(score_pair[0] - score_pair[1])
            ppl_key = 'correct' if min_ppl_id == correct_id else 'incorrect'
            perplexity_changes_pronoun[ppl_key].append(ppl_diff)

        # Track perplexity changes across samples
        if sample_id % 2 == 1:
            if len(contrastive_pair_ppl) == 2:
                if (contrastive_pair_ppl[0][0] is True) and (contrastive_pair_ppl[1][0] is True):
                    ppl_key = 'both_right'
                elif (contrastive_pair_ppl[0][0] is False) and (contrastive_pair_ppl[1][0] is False):
                    ppl_key = 'both_wrong'
                else:
                    ppl_key = 'same_choice'
                perplexity_changes_trigger[ppl_key].append(
                    abs(contrastive_pair_ppl[0][1] - contrastive_pair_ppl[1][1]))
            contrastive_pair_ppl = list()

        # Update pair counts
        if sample_id % 2 == 1:
            if 0 not in contrastive_pair_successes:
                num_correct_pairs += 1
            else:
                num_incorrect_pairs += 1
            contrastive_pair_successes = list()

    # Report
    out_dir = '/'.join(scores_path.split('/')[:-1])
    sorted_gen = sorted(list(PRON2GEN.values()))
    gen2num = {p: p_id + 1 for p_id, p in enumerate(sorted_gen)}
    print('=' * 20)
    print('# of SAMPLES with lower perplexity assigned to the CORRECT option: {:d}'.format(num_correct))
    print('# of SAMPLES with lower perplexity assigned to the INCORRECT option: {:d}'.format(num_incorrect))
    print('# of PAIRS were resolved correctly: {:d}'.format(num_correct_pairs))
    print('# of PAIRS were resolved incorrectly: {:d}'.format(num_incorrect_pairs))
    print('Accuracy, SAMPLES: {:.4f}'.format(num_correct / (num_correct + num_incorrect)))
    print('Accuracy, PAIRS: {:.4f}'.format(num_correct_pairs / (num_correct_pairs + num_incorrect_pairs)))

    print('=' * 20)
    print('--- MODEL HEURISTICS ---')
    print('Total gender counts for CORRECT referents: male: {:d} | female: {:d} | neutral: {:d}'.format(
        gender_counts['reference_true'].get('male', 0), gender_counts['reference_true'].get('female', 0),
        gender_counts['reference_true'].get('neutral', 0)))
    print('Total gender counts for model-PREFERRED referents: male: {:d} | female: {:d} | neutral: {:d}'.format(
        gender_counts['model_preference'].get('male', 0), gender_counts['model_preference'].get('female', 0),
        gender_counts['model_preference'].get('neutral', 0)))
    print('Total gender counts for INCORRECT referents: male: {:d} | female: {:d} | neutral: {:d}'.format(
        gender_counts['reference_false'].get('male', 0), gender_counts['reference_false'].get('female', 0),
        gender_counts['reference_false'].get('neutral', 0)))
    print('Total gender counts for model-REJECTED referents: male: {:d} | female: {:d} | neutral: {:d}'.format(
        gender_counts['model_rejection'].get('male', 0), gender_counts['model_rejection'].get('female', 0),
        gender_counts['model_rejection'].get('neutral', 0)))
    # Compare correlation
    print('-' * 5)
    print('Gender correlations for references (MWU):')
    x, y = list(), list()
    for cat, var in [('reference_true', x), ('reference_false', y)]:
        for key in gender_counts[cat].keys():
            var += [gen2num[key]] * gender_counts[cat][key]
    print(mwu(x, y, tail='two-sided'))
    print('Gender correlations for model choices (MWU):')
    x, y = list(), list()
    for cat, var in [('model_preference', x), ('model_rejection', y)]:
        for key in gender_counts[cat].keys():
            var += [gen2num[key]] * gender_counts[cat][key]
    print(mwu(x, y, tail='two-sided'))
    # Plot
    make_bar_plot(gender_counts, 'target pronoun gender', 'frequency', sorted_gen, out_dir, 'gender_counts')

    print('-' * 10)
    print('Referent position for CORRECT referents: first {:d} | second {:d}'.format(
        position_counts['reference_true']['first'], position_counts['reference_true']['second']))
    print('Referent position for model-PREFERRED references: first {:d} | second {:d}'.format(
        position_counts['model_preference']['first'], position_counts['model_preference']['second']))
    print('Referent position for INCORRECT referents: first {:d} | second {:d}'.format(
        position_counts['reference_false']['first'], position_counts['reference_false']['second']))
    print('Referent position for model-REJECTED references: first {:d} | second {:d}'.format(
        position_counts['model_rejection']['first'], position_counts['model_rejection']['second']))
    print('   Referent gender per referent position:')
    for loc in ref_gender_counts_per_position.keys():
        print('   {:s}:'.format(loc))
        for gen in ref_gender_counts_per_position[loc].keys():
            print('      {:s} : {:d}'.format(gen, ref_gender_counts_per_position[loc][gen]))
    # Compare correlation
    print('-' * 5)
    print('Position correlations for references (MWU):')
    x, y = list(), list()
    for cat, var in [('reference_true', x), ('reference_false', y)]:
        for key in position_counts[cat].keys():
            var += [int(key == 'first') + 1] * position_counts[cat][key]
    print(mwu(x, y, tail='two-sided'))
    print('Position correlations for model choices (MWU):')
    x, y = list(), list()
    for cat, var in [('model_preference', x), ('model_rejection', y)]:
        for key in position_counts[cat].keys():
            var += [int(key == 'first') + 1] * position_counts[cat][key]
    print(mwu(x, y, tail='two-sided'))
    # Plot
    make_bar_plot(
        position_counts, 'relative referent position', 'frequency', ['first', 'second'], out_dir, 'position_counts')

    print('-' * 10)
    print('# of sub-words between the pronoun and its referent for the CORRECT source interpretation:')
    flattened_list = list()
    for key in token_distance['reference_true'].keys():
        flattened_list += [key] * token_distance['reference_true'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    print('-' * 5)
    print('# of sub-words between the pronoun and its referent for the interpretation PREFERRED by the model:')
    flattened_list = list()
    for key in token_distance['model_preference'].keys():
        flattened_list += [key] * token_distance['model_preference'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    print('-' * 5)
    print('# of sub-words between the pronoun and its referent for the INCORRECT source interpretation:')
    flattened_list = list()
    for key in token_distance['reference_false'].keys():
        flattened_list += [key] * token_distance['reference_false'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    print('-' * 5)
    print('# of sub-words between the pronoun and its referent for the interpretation REJECTED by the model:')
    flattened_list = list()
    for key in token_distance['model_rejection'].keys():
        flattened_list += [key] * token_distance['model_rejection'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    # Compare correlation
    print('-' * 5)
    print('Distance correlations for references (MWU):')
    x, y = list(), list()
    for cat, var in [('reference_true', x), ('reference_false', y)]:
        for key in token_distance[cat].keys():
            var += [key] * token_distance[cat][key]
    print(mwu(x, y, tail='two-sided'))
    print('Distance correlations for model choices (MWU):')
    x, y = list(), list()
    for cat, var in [('model_preference', x), ('model_rejection', y)]:
        for key in token_distance[cat].keys():
            var += [key] * token_distance[cat][key]
    print(mwu(x, y, tail='two-sided'))
    # Plot
    token_distance_plot = {'reference_true': token_distance['reference_true'],
                           'model_preference': token_distance['model_preference']}
    make_density_plot(token_distance_plot, ['category', '# sub-words'], out_dir, 'token_distance')

    # ACROSS both samples in each PAIR
    print('-' * 10)
    print('Perplexity difference when triggers are ignored (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(perplexity_changes_trigger['same_choice']), np.mean(perplexity_changes_trigger['same_choice']),
        np.std(perplexity_changes_trigger['same_choice'])))
    print('Perplexity difference when both predictions are correct (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(perplexity_changes_trigger['both_right']),
        np.mean(perplexity_changes_trigger['both_right']), np.std(perplexity_changes_trigger['both_right'])))
    print('Perplexity difference when both predictions are incorrect (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(perplexity_changes_trigger['both_wrong']), np.mean(perplexity_changes_trigger['both_wrong']),
        np.std(perplexity_changes_trigger['both_wrong'])))
    # Compare correlations
    print('MWU: same_choice - both_right')
    print(mwu(perplexity_changes_trigger['same_choice'], perplexity_changes_trigger['both_right'], tail='two-sided'))
    print('MWU: same_choice - both_wrong')
    print(mwu(perplexity_changes_trigger['same_choice'], perplexity_changes_trigger['both_wrong'], tail='two-sided'))
    print('MWU: both_right - both_wrong')
    print(mwu(perplexity_changes_trigger['both_right'], perplexity_changes_trigger['both_wrong'], tail='two-sided'))
    # Plot
    make_density_plot(perplexity_changes_trigger, ['category', 'PPl difference'], out_dir, 'ppl_difference_trigger')

    # WITHIN each SAMPLE
    print('-' * 10)
    print('Perplexity difference for correct translation choice (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(perplexity_changes_pronoun['correct']), np.mean(perplexity_changes_pronoun['correct']),
        np.std(perplexity_changes_pronoun['correct'])))
    print('Perplexity difference for incorrect translation choice (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(perplexity_changes_pronoun['incorrect']), np.mean(perplexity_changes_pronoun['incorrect']),
        np.std(perplexity_changes_pronoun['incorrect'])))
    # Compare correlations
    print('MWU: correct - incorrect')
    print(mwu(perplexity_changes_pronoun['correct'], perplexity_changes_pronoun['incorrect'], tail='two-sided'))
    # Plot
    make_density_plot(perplexity_changes_pronoun, ['category', 'PPl difference'], out_dir, 'ppl_difference_pronoun')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='path to the JSON file containing challenge set samples')
    parser.add_argument('--scores_path', type=str, required=True,
                        help='path to the text file containing model scores')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='path to the text file containing reference labels')
    parser.add_argument('--codes_path', type=str, required=True,
                        help='path to the files containing the (source) BPE-codes of the evaluated model')
    parser.add_argument('--src_vocab_path', type=str, default=None,
                        help='path to the file containing model (source) vocabulary')
    parser.add_argument('--src_lang', type=str, required=True, choices=['en'],
                        help='code corresponding to the target language of the translations')
    parser.add_argument('--tgt_lang', type=str, required=True, choices=['de', 'fr', 'ru'],
                        help='code corresponding to the target language of the translations')
    parser.add_argument('--use_subword_nmt', action='store_true',
                        help='set to TRUE when evaluating the WMT14 en-fr model')
    args = parser.parse_args()

    # Select relevant pronouns
    PRON2GEN = LANG2PRON[args.tgt_lang]

    # Initialize BPE
    if args.use_subword_nmt:
        with open(args.src_vocab_path, 'r', encoding='utf8') as sv:
            vocab = read_vocabulary(sv, 50)
        bpe_model = None
    else:
        vocab = None
        bpe_model = fastBPE.fastBPE(args.codes_path, args.src_vocab_path)

    # Initialize tokenizer
    src_mt = MosesTokenizer(args.src_lang)

    compute_accuracy(args.json_file_path, args.scores_path, args.labels_path, args.codes_path, args.tgt_lang)

