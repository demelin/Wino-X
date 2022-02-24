import os
import math
import torch
import string
import stanza
import argparse

import numpy as np

from pingouin import mwu
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, XLMRobertaTokenizer, \
    RobertaForMaskedLM, BartForConditionalGeneration, BartTokenizer, MBartTokenizer, MBartForConditionalGeneration

from eval_util import read_jsonl, get_final_index, count_distance_in_subwords, make_bar_plot, make_density_plot


LANG2PRON = {'de': {'er': 'male', 'sie': 'female', 'es': 'neutral'},
             'fr': {'il': 'male', 'elle': 'female', 'ils': 'male', 'elles': 'female'},
             'ru': {'oн': 'male', 'oнa': 'female', 'oнo': 'neutral'}}

GEN_ABBR = {'Masc': 'male',
            'Fem': 'female',
            'Neut': 'neutral'}

MBART_TABLE = {'en': 'en_XX',
               'de': 'de_DE',
               'fr': 'fr_XX',
               'ru': 'ru_RU'}


def _score_sequence_masked(context, filler):
    """ Assigns an MLM score to the filled-in context sentence, based on pseudo log-likelihood. """

    # Fill-in context
    sequence = context.replace('_', filler)

    # Tokenize
    non_special_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence)))
    sequence_ids = tokenizer.encode(sequence)

    sequence_token_probs = list()
    for i_id, i in enumerate(sequence_ids):
        if i not in non_special_ids:
            continue
        # Make copies
        masked_sequence_ids = sequence_ids[:]
        # Mask
        masked_sequence_ids[i_id] = tokenizer.mask_token_id
        masked_sequence_tensor = torch.tensor([masked_sequence_ids]).to(device)
        # Estimate pseudo-likelihood of sequence
        with torch.no_grad():
            token_logits = model(masked_sequence_tensor)[0]

        masked_logits = token_logits[0, i_id, :]
        masked_softmax = torch.softmax(masked_logits, dim=0)
        # Get the filler probability
        sequence_token_probs.append(np.log(masked_softmax[i].item()))

    # filler_probability = np.exp(sum([np.log(s) for s in filler_probs]))
    sequence_pppl = np.exp(-np.mean(sequence_token_probs))
    return sequence_pppl


def _score_sequence_generative(context, filler):
    """ Computes perplexity of a filled-in context sequence containing the specified filler """

    # Score target sequence conditioned on the input
    if args.model_type == 'bart':
        # Create model-appropriate input and target
        src_sequence = context.replace('_', tokenizer.mask_token)  # span masking requires only one <mask>
        tgt_sequence = context.replace('_', filler)

        if args.use_multi:
            input_batch = tokenizer.prepare_seq2seq_batch(src_sequence,
                                                          src_lang=MBART_TABLE[args.tgt_lang],
                                                          tgt_lang=MBART_TABLE[args.tgt_lang],
                                                          tgt_texts=tgt_sequence, return_tensors='pt')
        else:
            input_batch = tokenizer.prepare_seq2seq_batch(src_sequence, tgt_texts=tgt_sequence, return_tensors='pt')
    else:
        # Create model-appropriate input and target
        src_sequence = context.replace('_', '<extra_id_0>')
        tgt_sequence = '<extra_id_0> {:s} <extra_id_1>'.format(filler)

        input_batch = \
            tokenizer.prepare_seq2seq_batch(src_texts=src_sequence, tgt_texts=tgt_sequence, return_tensors='pt')

    with torch.no_grad():
        loss = model(input_ids=input_batch['input_ids'].to(device),
                     labels=input_batch['labels'].to(device)).loss.item()

    return math.exp(loss), src_sequence, tgt_sequence  # return target sequence perplexity


def evaluate_models(json_file_path, out_dir, src_lang, tgt_lang):
    """ Checks model performance on the challenge data. """

    # Read-in data
    samples = read_jsonl(json_file_path)

    # Derive MLM samples from contra samples
    if 'contra' in json_file_path:
        mlm_samples = list()
        for s in samples:
            mlm_s = dict()
            tgt_context = s['translation1']
            mlm_s['option1'] = s['pronoun1']
            mlm_s['option2'] = s['pronoun2']
            mlm_s['answer'] = s['answer']
            mlm_s['context_referent_of_option1_de'] = s['true_translation_referent_of_pronoun1_de']
            mlm_s['context_referent_of_option2_de'] = s['true_translation_referent_of_pronoun2_de']
            # Introduce gap
            context_tokens = tgt_context.split()
            prn_pos = get_final_index(context_tokens, s['pronoun1'])
            context_tokens[prn_pos] = '_'
            mlm_s['tgt_context'] = ' '.join(context_tokens)
            mlm_samples.append(mlm_s)
        samples = mlm_samples

    # Compute accuracy and track input features that may be indicative of the model using heuristics
    num_correct, num_incorrect, num_correct_pairs, num_incorrect_pairs = 0, 0, 0, 0

    gender_counts = None
    ref_gender_counts_per_position = None
    if tgt_lang != 'en':
        gender_counts = {'reference_true': {v: 0 for v in PRON2GEN.values()},
                         'reference_false': {v: 0 for v in PRON2GEN.values()},
                         'model_preference': {v: 0 for v in PRON2GEN.values()},
                         'model_rejection': {v: 0 for v in PRON2GEN.values()}}

        ref_gender_counts_per_position = {'first': {v: 0 for v in PRON2GEN.values()},
                                          'second': {v: 0 for v in PRON2GEN.values()}}  # for correct referents only

    position_counts = {'reference_true': {'first': 0, 'second': 0},
                       'reference_false': {'first': 0, 'second': 0},
                       'model_preference': {'first': 0, 'second': 0},
                       'model_rejection': {'first': 0, 'second': 0}}

    token_distance = \
        {'reference_true': dict(), 'reference_false': dict(), 'model_preference': dict(), 'model_rejection': dict()}
    # This is intended to estimate the "influence" of the trigger phrases
    score_changes_trigger = {'same_choice': list(), 'both_right': list(), 'both_wrong': list()}
    # This is intended to estimate the "influence" of the target pronoun
    score_changes_pronoun = {'correct': list(), 'incorrect': list()}

    contrastive_pair_score = list()
    contrastive_pair_successes = list()
    for sample_id, sample in enumerate(samples):

        # Report
        print('Checking sample {:d}'.format(sample_id))
        if (sample_id - 1) > 0 and (sample_id - 1) % 100 == 0:
            print('Processed {:d} samples'.format(sample_id - 1))

        context = sample['context_{:s}'.format(src_lang)] if has_noun_fillers else sample['tgt_context']
        option1 = sample['option1_{:s}'.format(tgt_lang)] if has_noun_fillers else sample['option1']
        option2 = sample['option2_{:s}'.format(tgt_lang)] if has_noun_fillers else sample['option2']
        if args.model_type == 'bart':
            option1_score = _score_sequence_generative(context, option1)
            option2_score = _score_sequence_generative(context, option2)
        else:
            option1_score = _score_sequence_masked(context, option1)
            option2_score = _score_sequence_masked(context, option2)
        score_pair = [option1_score, option2_score]
        label_pair = [int(int(sample['answer']) == 1), int(int(sample['answer']) == 2)]

        # Identify filler gender for both options
        gen_pair, true_filler_gen, false_filler_gen = None, None, None
        if tgt_lang != 'en':
            if not has_noun_fillers:
                # Identify filler gender in both translations
                try:
                    gen_pair = [PRON2GEN[option1], PRON2GEN[option2]]
                except KeyError:
                    gen_pair = [None, None]
            else:
                gen_pair = list()
                for option in [option1, option2]:
                    # Parse options with Stanza to identify their gender
                    parsed_option = nlp(option).sentences[0].words
                    for word in parsed_option:
                        try:
                            word_feats = \
                                {f.split('=')[0].strip(): f.split('=')[1].strip() for f in word.feats.split('|')}
                            option_gen = word_feats.get('Gender', None)
                            if option_gen is not None:
                                # Prioritize determiner gender, for languages with determiners
                                gen_pair.append(GEN_ABBR[option_gen])
                                break
                        except AttributeError or KeyError:
                            continue

            true_filler_gen = gen_pair[label_pair.index(max(label_pair))]
            false_filler_gen = gen_pair[label_pair.index(min(label_pair))]

        # Establish the relative position of the correct referent
        if src_lang == 'en':
            assert has_noun_fillers is True, 'Pronoun-MLM evaluation is not compatible with English'
            referent1 = sample['option1_en']
            referent2 = sample['option2_en']
        else:
            referent1 = sample['context_referent_of_option1_{:s}'.format(src_lang)].strip(string.punctuation)  # strip added due to bug
            referent2 = sample['context_referent_of_option2_{:s}'.format(src_lang)].strip(string.punctuation)

        context_tokens = [tok.strip(string.punctuation) for tok in context.split()]
        true_referent_is_first, distance_pair = None, None
        try:
            if int(sample['answer']) == 1:
                true_referent_is_first = get_final_index(context_tokens, referent1) < \
                                         get_final_index(context_tokens, referent2)
            else:
                true_referent_is_first = get_final_index(context_tokens, referent2) < \
                                         get_final_index(context_tokens, referent1)

            if true_referent_is_first:
                position_counts['reference_true']['first'] += 1
                position_counts['reference_false']['second'] += 1
                if true_filler_gen is not None:
                    ref_gender_counts_per_position['first'][true_filler_gen] += 1
                    ref_gender_counts_per_position['second'][false_filler_gen] += 1
            else:
                position_counts['reference_true']['second'] += 1
                position_counts['reference_false']['first'] += 1
                if true_filler_gen is not None:
                    ref_gender_counts_per_position['first'][false_filler_gen] += 1
                    ref_gender_counts_per_position['second'][true_filler_gen] += 1

            # Measure sub-word distance
            distance_pair = [count_distance_in_subwords(context, referent1, '_', None, None, None, tokenizer),
                             count_distance_in_subwords(context, referent2, '_', None, None, None, tokenizer)]
        except AssertionError:
            pass

        # Check model preferences
        if score_pair[0] == score_pair[1]:
            num_incorrect += 1
            contrastive_pair_successes.append(0)
        else:
            min_score_id = score_pair.index(min(score_pair))  # lower pppl is better
            correct_id = label_pair.index(max(label_pair))

            # Track model accuracy
            if min_score_id == correct_id:
                num_correct += 1
                contrastive_pair_successes.append(1)
            else:
                num_incorrect += 1
                contrastive_pair_successes.append(0)

            # Track model scores for the preferred filler
            contrastive_pair_score.append((min_score_id == correct_id, score_pair[min_score_id]))

            # Track heuristics
            # Gender
            if true_filler_gen is not None:
                gender_counts['reference_true'][gen_pair[correct_id]] += 1
                gender_counts['reference_false'][gen_pair[abs(1 - correct_id)]] += 1
                gender_counts['model_preference'][gen_pair[min_score_id]] += 1
                gender_counts['model_rejection'][gen_pair[abs(1 - min_score_id)]] += 1

            # Relative referent position
            if true_referent_is_first is not None:
                if true_referent_is_first:
                    pref_key, rej_key = ('first', 'second') if min_score_id == correct_id else ('second', 'first')
                else:
                    pref_key, rej_key = ('second', 'first') if min_score_id == correct_id else ('first', 'second')
                position_counts['model_preference'][pref_key] += 1
                position_counts['model_rejection'][rej_key] += 1

            # Sub-word distance
            if distance_pair is not None:
                for key, tpl_id in [('reference_true', correct_id), ('reference_false', abs(1 - correct_id)),
                                    ('model_preference', min_score_id), ('model_rejection', abs(1 - min_score_id))]:
                    if token_distance[key].get(distance_pair[tpl_id], None) is None:
                        token_distance[key][distance_pair[tpl_id]] = 0
                    token_distance[key][distance_pair[tpl_id]] += 1

            # Track score changes within the sample
            score_diff = abs(score_pair[0] - score_pair[1])
            score_key = 'correct' if min_score_id == correct_id else 'incorrect'
            score_changes_pronoun[score_key].append(score_diff)

        # Track score changes across samples
        if sample_id % 2 == 1:
            if len(contrastive_pair_score) == 2:
                if (contrastive_pair_score[0][0] is True) and (contrastive_pair_score[1][0] is True):
                    score_key = 'both_right'
                elif (contrastive_pair_score[0][0] is False) and (contrastive_pair_score[1][0] is False):
                    score_key = 'both_wrong'
                else:
                    score_key = 'same_choice'
                score_changes_trigger[score_key].append(abs(contrastive_pair_score[0][1] -
                                                            contrastive_pair_score[1][1]))
            contrastive_pair_score = list()

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
    print('=' * 20)
    print('# of SAMPLES with lower perplexity assigned to the CORRECT option: {:d}'.format(num_correct))
    print('# of SAMPLES with lower perplexity assigned to the INCORRECT option: {:d}'.format(num_incorrect))
    print('# of PAIRS were resolved correctly: {:d}'.format(num_correct_pairs))
    print('# of PAIRS were resolved incorrectly: {:d}'.format(num_incorrect_pairs))
    print('Accuracy, SAMPLES: {:.4f}'.format(num_correct / (num_correct + num_incorrect)))
    print('Accuracy, PAIRS: {:.4f}'.format(num_correct_pairs / (num_correct_pairs + num_incorrect_pairs)))

    if tgt_lang != 'en':
        sorted_gen = sorted(list(PRON2GEN.values()))
        gen2num = {p: p_id + 1 for p_id, p in enumerate(sorted_gen)}
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
    if tgt_lang != 'en':
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
    print('# of sub-words between the gap and its referent for the CORRECT interpretation:')
    flattened_list = list()
    for key in token_distance['reference_true'].keys():
        flattened_list += [key] * token_distance['reference_true'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    print('-' * 5)
    print('# of sub-words between the gap and its referent for the interpretation PREFERRED by the model:')
    flattened_list = list()
    for key in token_distance['model_preference'].keys():
        flattened_list += [key] * token_distance['model_preference'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    print('-' * 5)
    print('# of sub-words between the gap and its referent for the INCORRECT interpretation:')
    flattened_list = list()
    for key in token_distance['reference_false'].keys():
        flattened_list += [key] * token_distance['reference_false'][key]
    print('mean: {:.4f} | std: {:.4f}'.format(np.mean([flattened_list]), np.std(flattened_list)))
    print('-' * 5)
    print('# of sub-words between the gap and its referent for the interpretation REJECTED by the model:')
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
    print('Score difference when triggers are ignored (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(score_changes_trigger['same_choice']), np.mean(score_changes_trigger['same_choice']),
        np.std(score_changes_trigger['same_choice'])))
    print('Score difference when both predictions are correct (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(score_changes_trigger['both_right']),
        np.mean(score_changes_trigger['both_right']), np.std(score_changes_trigger['both_right'])))
    print('Score difference when both predictions are incorrect (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(score_changes_trigger['both_wrong']), np.mean(score_changes_trigger['both_wrong']),
        np.std(score_changes_trigger['both_wrong'])))
    # Compare correlations
    print('MWU: same_choice - both_right')
    print(mwu(score_changes_trigger['same_choice'], score_changes_trigger['both_right'], tail='two-sided'))
    print('MWU: same_choice - both_wrong')
    print(mwu(score_changes_trigger['same_choice'], score_changes_trigger['both_wrong'], tail='two-sided'))
    print('MWU: both_right - both_wrong')
    print(mwu(score_changes_trigger['both_right'], score_changes_trigger['both_wrong'], tail='two-sided'))
    # Plot
    make_density_plot(score_changes_trigger, ['category', 'PPl difference'], out_dir, 'ppl_difference_trigger')

    # WITHIN each SAMPLE
    print('-' * 10)
    print('Score difference for correct referent choice (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
        len(score_changes_pronoun['correct']), np.mean(score_changes_pronoun['correct']),
        np.std(score_changes_pronoun['correct'])))
    print('Score difference for incorrect referent choice (count, mean, std): {:d} | {:.4f} | {:.4f}'.format(
            len(score_changes_pronoun['incorrect']), np.mean(score_changes_pronoun['incorrect']),
            np.std(score_changes_pronoun['incorrect'])))
    # Compare correlations
    print('MWU: correct - incorrect')
    print(mwu(score_changes_pronoun['correct'], score_changes_pronoun['incorrect'], tail='two-sided'))
    # Plot
    make_density_plot(score_changes_pronoun, ['category', 'PPl difference'], out_dir, 'ppl_difference_pronoun')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='path to the JSON file containing the contrastive samples')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='path to the directory containing checkpoint files of the evaluated model')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--model_type', type=str, choices=['bert', 'roberta', 'bart'],
                        help='Model type to evaluate')
    parser.add_argument('--use_cpu', action='store_true', help='Whether to use the CPU')
    parser.add_argument('--src_lang', type=str, choices=['en', 'de', 'fr', 'ru'],
                        help='Source language of the challenge set')
    parser.add_argument('--tgt_lang', type=str, choices=['en', 'de', 'fr', 'ru'],
                        help='Target language of the challenge set')
    parser.add_argument('--use_multi', action='store_true',
                        help='Whether to use a multilingual language model; if disabled, uses a monolingual target '
                             'language model')
    parser.add_argument('--use_xlm_base', action='store_true', help='Whether to use a XML-R BASE')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Check format
    has_noun_fillers = 'mlm_nouns' in args.json_file_path

    # Select relevant pronouns
    if args.tgt_lang != 'en':
        PRON2GEN = LANG2PRON[args.tgt_lang]

    # Assign checkpoints (change 'base' to 'large', if need be)
    if args.model_type == 'roberta':
        model_type = RobertaForMaskedLM
        tokenizer_type = AutoTokenizer
        if args.use_multi:
            tokenizer_type = XLMRobertaTokenizer
            if args.use_xlm_base:
                checkpoint = 'xlm-roberta-base'
            else:
                checkpoint = 'xlm-roberta-large'
        else:
            assert args.tgt_lang == 'en', 'Configuration not supported!'
            checkpoint = 'roberta-large'
    elif args.model_type == 'bart':
        if args.use_multi:
            model_type = BartForConditionalGeneration
            tokenizer_type = BartTokenizer
            checkpoint = 'facebook/mbart-large-cc25'
        else:
            assert args.tgt_lang == 'en', 'Configuration not supported!'
            model_type = MBartForConditionalGeneration
            tokenizer_type = MBartTokenizer
            checkpoint = 'facebook/bart-large'
    else:
        if args.use_multi:
            model_type = BertForMaskedLM
            tokenizer_type = BertTokenizer
            checkpoint = 'bert-base-multilingual-cased'
        else:
            if args.tgt_lang == 'en':
                model_type = BertForMaskedLM
                tokenizer_type = BertTokenizer
                checkpoint = 'bert-base-cased'
            elif args.tgt_lang == 'de':
                model_type = BertForMaskedLM
                tokenizer_type = BertTokenizer
                checkpoint = 'bert-base-german-dbmdz-cased'
            elif args.tgt_lang == 'fr':
                model_type = RobertaForMaskedLM
                tokenizer_type = AutoTokenizer
                checkpoint = 'camembert-base'
            else:
                model_type = BertForMaskedLM
                tokenizer_type = BertTokenizer
                checkpoint = args.checkpoint_dir  # DeepPavlov model

    if args.checkpoint_dir is not None:
        checkpoint = args.checkpoint_dir

    # Load models and tokenizers
    print('Chosen checkpoint : ', checkpoint)
    tokenizer = tokenizer_type.from_pretrained(checkpoint)
    model = model_type.from_pretrained(checkpoint)

    device = 'cpu' if args.use_cpu else 'cuda:0'
    model.to(device)

    # Initialize Stanza models
    if args.tgt_lang != 'ru':
        nlp = stanza.Pipeline(args.tgt_lang, processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)
    else:
        nlp = stanza.Pipeline(args.tgt_lang, processors='tokenize,pos,lemma,depparse', use_gpu=True)

    evaluate_models(args.json_file_path, args.out_dir,  args.src_lang, args.tgt_lang)

