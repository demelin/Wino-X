import os
import torch
import string
import argparse

import numpy as np

from pingouin import mwu
from transformers import (BartForConditionalGeneration,
                          BartTokenizer,
                          MBartTokenizer,
                          MBart50TokenizerFast,
                          MBartForConditionalGeneration)

from eval_util import read_jsonl, get_final_index, compute_perplexity, count_distance_in_subwords, make_bar_plot, \
    make_density_plot

MBART_TABLE = {'en': 'en_XX',
               'de': 'de_DE',
               'fr': 'fr_FR',
               'ru': 'ru_RU'}

LANG2PRON = {'de': {'er': 'male', 'sie': 'female', 'es': 'neutral'},
             'fr': {'il': 'male', 'elle': 'female', 'ils': 'male', 'elles': 'female'},
             'ru': {'oн': 'male', 'oнa': 'female', 'oнo': 'neutral'}}


def _score_translation(src_sentence, tgt_sentence):
    """ Computes perplexity of contrastive translations """
    # Score target sequence conditioned on the input
    src = tokenizer(src_sentence, return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        tgt = tokenizer(tgt_sentence, return_tensors='pt').input_ids

    # Do a forward pass
    with torch.no_grad():
        model_out = model(input_ids=src.input_ids.to(device),
                          attention_mask=src.attention_mask.to(device),
                          labels=tgt.to(device))
    # Extract probabilities
    tgt_input_ids = torch.squeeze(tgt).detach().cpu().numpy().tolist()
    model_logits = torch.squeeze(model_out.logits)
    tgt_probs = model_logits.softmax(axis=-1).detach().cpu().numpy().tolist()
    token_probabilities = [tgt_probs[probs_row][tok_id] for probs_row, tok_id in enumerate(tgt_input_ids)]
    return compute_perplexity(token_probabilities)


def evaluate_models(json_file_path, out_dir):
    """ Checks model performance on the challenge data. """

    # Read-in data
    samples = read_jsonl(json_file_path)

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
    ppl_changes_trigger = {'same_choice': list(), 'both_right': list(), 'both_wrong': list()}
    # This is intended to estimate the "influence" of the target pronoun
    ppl_changes_pronoun = {'correct': list(), 'incorrect': list()}

    contrastive_pair_ppl = list()
    contrastive_pair_successes = list()
    for sample_id, sample in enumerate(samples):

        # Report
        print('Checking sample {:d}'.format(sample_id))
        if (sample_id - 1) > 0 and (sample_id - 1) % 100 == 0:
            print('Processed {:d} samples'.format(sample_id - 1))

        src = sample['sentence']
        tgt1 = sample['translation1']
        tgt2 = sample['translation2']
        tgt1_ppl = _score_translation(src, tgt1)
        tgt2_ppl = _score_translation(src, tgt2)
        score_pair = [tgt1_ppl, tgt2_ppl]
        label_pair = [int(int(sample['answer']) == 1), int(int(sample['answer']) == 2)]

        # Identify filler gender in both translations
        try:
            gen_pair = [PRON2GEN[samples[sample_id]['pronoun1']], PRON2GEN[samples[sample_id]['pronoun2']]]
        except KeyError:
            gen_pair = [None, None]
        true_ref_gen = gen_pair[label_pair.index(max(label_pair))]
        false_ref_gen = gen_pair[label_pair.index(min(label_pair))]

        # Establish the relative position of the correct referent
        sentence_tokens = [tok.strip(string.punctuation) for tok in sample['sentence'].split()]
        if int(sample['answer']) == 1:
            true_referent_is_first = get_final_index(sentence_tokens, sample['referent1_en']) < \
                                     get_final_index(sentence_tokens, sample['referent2_en'])
        else:
            true_referent_is_first = get_final_index(sentence_tokens, sample['referent2_en']) < \
                                     get_final_index(sentence_tokens, sample['referent1_en'])
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
        distance_pair = \
            [count_distance_in_subwords(sample['sentence'], sample['referent1_en'], 'it', None, None, None, tokenizer),
             count_distance_in_subwords(sample['sentence'], sample['referent2_en'], 'it', None, None, None, tokenizer)]

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

            # Measure sub-word distance
            for key, tpl_id in [('reference_true', correct_id), ('reference_false', abs(1 - correct_id)),
                                ('model_preference', min_ppl_id), ('model_rejection', abs(1 - min_ppl_id))]:
                if token_distance[key].get(distance_pair[tpl_id], None) is None:
                    token_distance[key][distance_pair[tpl_id]] = 0
                token_distance[key][distance_pair[tpl_id]] += 1

            # Track perplexity changes within the sample
            ppl_diff = abs(score_pair[0] - score_pair[1])
            ppl_key = 'correct' if min_ppl_id == correct_id else 'incorrect'
            ppl_changes_pronoun[ppl_key].append(ppl_diff)

        # Track perplexity changes across samples
        if sample_id % 2 == 1:
            if len(contrastive_pair_ppl) == 2:
                if (contrastive_pair_ppl[0][0] is True) and (contrastive_pair_ppl[1][0] is True):
                    ppl_key = 'both_right'
                elif (contrastive_pair_ppl[0][0] is False) and (contrastive_pair_ppl[1][0] is False):
                    ppl_key = 'both_wrong'
                else:
                    ppl_key = 'same_choice'
                ppl_changes_trigger[ppl_key].append(
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
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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
        len(ppl_changes_trigger['same_choice']), np.mean(ppl_changes_trigger['same_choice']),
        np.std(ppl_changes_trigger['same_choice'])))
    print('Perplexity difference when both predictions are correct (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(ppl_changes_trigger['both_right']),
        np.mean(ppl_changes_trigger['both_right']), np.std(ppl_changes_trigger['both_right'])))
    print('Perplexity difference when both predictions are incorrect (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(ppl_changes_trigger['both_wrong']), np.mean(ppl_changes_trigger['both_wrong']),
        np.std(ppl_changes_trigger['both_wrong'])))
    # Compare correlations
    print('MWU: same_choice - both_right')
    print(mwu(ppl_changes_trigger['same_choice'], ppl_changes_trigger['both_right'], tail='two-sided'))
    print('MWU: same_choice - both_wrong')
    print(mwu(ppl_changes_trigger['same_choice'], ppl_changes_trigger['both_wrong'], tail='two-sided'))
    print('MWU: both_right - both_wrong')
    print(mwu(ppl_changes_trigger['both_right'], ppl_changes_trigger['both_wrong'], tail='two-sided'))
    # Plot
    make_density_plot(ppl_changes_trigger, ['category', 'PPl difference'], out_dir, 'ppl_difference_trigger')

    # WITHIN each SAMPLE
    print('-' * 10)
    print('Perplexity difference for correct translation choice (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(ppl_changes_pronoun['correct']), np.mean(ppl_changes_pronoun['correct']),
        np.std(ppl_changes_pronoun['correct'])))
    print('Perplexity difference for incorrect translation choice (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(ppl_changes_pronoun['incorrect']), np.mean(ppl_changes_pronoun['incorrect']),
        np.std(ppl_changes_pronoun['incorrect'])))
    # Compare correlations
    print('MWU: correct - incorrect')
    print(mwu(ppl_changes_pronoun['correct'], ppl_changes_pronoun['incorrect'], tail='two-sided'))
    # Plot
    make_density_plot(ppl_changes_pronoun, ['category', 'PPl difference'], out_dir, 'ppl_difference_pronoun')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='path to the JSON file containing the contrastive samples')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='path to the directory containing checkpoint files of the evaluated model')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--model_type', type=str, choices=['bart', 'mbart50'],
                        help='Model type to evaluate')
    parser.add_argument('--use_multi', action='store_true',
                        help='Whether to use a multilingual language model; if disabled, uses a monolingual target '
                             'language model')
    parser.add_argument('--use_cpu', action='store_true', help='Whether to use the CPU')
    parser.add_argument('--tgt_lang', type=str, choices=['de', 'fr', 'ru'],
                        help='Target language of the challenge set; ONLY AFFECTS THE ANSWERS, as the context and '
                             'question are always presented in English')
    args = parser.parse_args()

    # Select relevant pronouns
    PRON2GEN = LANG2PRON[args.tgt_lang]

    # Assign checkpoints (change model size by changing checkpoint names)
    if args.model_type == 'bart':
        if args.use_multi:
            model_type = MBartForConditionalGeneration
            tokenizer_type = MBartTokenizer
        else:
            assert args.tgt_lang == 'en', 'Configuration not supported!'
            model_type = BartForConditionalGeneration
            tokenizer_type = BartTokenizer
    else:
        # mBART-50
        model_type = MBartForConditionalGeneration
        tokenizer_type = MBart50TokenizerFast

    # Load models and tokenizers
    if args.model_type != 'mbart50':
        assert args.checkpoint_dir is not None, 'Model checkpoint must be specified for models other than MBART50'
        tokenizer = tokenizer_type.from_pretrained(args.checkpoint_dir)
        model = model_type.from_pretrained(args.checkpoint_dir)
    else:
        model = model_type.from_pretrained('facebook/mbart-large-50-one-to-many-mmt')
        tokenizer = tokenizer_type.from_pretrained('facebook/mbart-large-50-one-to-many-mmt',
                                                   src_lang=MBART_TABLE['en'], tgt_lang=MBART_TABLE[args.tgt_lang])

    device = 'cpu' if args.use_cpu else 'cuda:0'
    model.to(device)

    evaluate_models(args.json_file_path, args.out_dir)
