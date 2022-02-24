import os
import json
import logging
import argparse

import mxnet as mx

from mlm.scorers import MLMScorer
from mlm.models import get_pretrained

from util import read_jsonl
from test_animacy import test_if_animate


def annotate_samples(json_path):
    """ Annotates WinoGrande samples with linguistic information that can be used for subsequent filtering. """

    # Read
    records = read_jsonl(json_path)

    # Pair
    paired_samples = dict()
    for s_iid, s in enumerate(records):
        s_id = s['qID'].split('-')[0]
        if not paired_samples.get(s_id, None):
            paired_samples[s_id] = [s]
        else:
            paired_samples[s_id].append(s)

    # Needed for animacy testing
    default_scores_dict = {'default_animate_score_sg': None,
                           'default_inanimate_score_sg': None,
                           'default_animal_score_sg': None,
                           'default_animate_score_pl': None,
                           'default_inanimate_score_pl': None,
                           'default_animal_score_pl': None}

    filter_pass = {'kept': dict(),
                   'animate_ref': dict(),
                   'animal_ref': dict()}

    for s_iid, s_id in enumerate(paired_samples.keys()):
        pair = paired_samples[s_id]
        option1 = pair[0]['option1'].lower()
        option2 = pair[0]['option2'].lower()
        s1_sentence = pair[0]['filled_sentence']
        s2_sentence = pair[1]['filled_sentence']
        s1_ambiguous_sentence = pair[0]['ambiguous_sentence']
        s2_ambiguous_sentence = pair[1]['ambiguous_sentence']

        logging.warning('+' * 20)
        logging.warning('Checking new pair ...')
        true_referent1 = pair[0]['option1'] if pair[0]['answer'] == '1' else pair[0]['option2']
        logging.warning('{}, {}'.format(s1_sentence, true_referent1))
        logging.warning(s1_ambiguous_sentence)
        logging.warning('-' * 5)
        true_referent2 = pair[1]['option1'] if pair[1]['answer'] == '1' else pair[1]['option2']
        logging.warning('{}, {}'.format(s2_sentence, true_referent2))
        logging.warning(s2_ambiguous_sentence)
        logging.warning('+' * 20)

        # Check if referents are animate
        option1_out = test_if_animate(option1, pair[0]['referent1_is_singular'], scorer, default_scores_dict)
        default_scores_dict = option1_out[2]
        option2_out = test_if_animate(option2, pair[0]['referent2_is_singular'], scorer, default_scores_dict)
        default_scores_dict = option1_out[2]
        option1_is_animate, option1_is_animal = option1_out[:2]
        option2_is_animate, option2_is_animal = option2_out[:2]

        if option1_is_animal or option2_is_animal:
            filter_pass['animal_ref'][s_id] = paired_samples[s_id]

        if (option1_is_animate and pair[0]['referent1_is_singular'] and not option1_is_animal) or \
                (option2_is_animate and pair[0]['referent1_is_singular'] and not option2_is_animal):

            filter_pass['animate_ref'][s_id] = paired_samples[s_id]

            logging.warning('=' * 20)
            logging.warning('Referent is animate or is animal:')
            logging.warning('Option 1: {} | is animate: {} | is animal: {}'
                            .format(option1, option1_is_animate, option1_is_animal))
            logging.warning('Option 2: {} | is animate: {} | is animal: {}'
                            .format(option2, option2_is_animate, option2_is_animal))

        else:
            filter_pass['kept'][s_id] = paired_samples[s_id]

        # Report periodically
        print('Seen {:d} / {:d} samples'.format(s_iid, len(paired_samples.keys())))
        if s_iid > 0 and (s_iid + 1) % 100 == 0:
            logging.warning('\n\n')
            logging.warning('-' * 20)
            logging.warning('Annotated and evaluated {:d} WinoGrande samples'.format(s_iid + 1))
            logging.warning('Samples kept: {:d}'.format(len(filter_pass['kept'])))
            logging.warning('Samples dropped: {:d}\n\n'.format((s_iid + 1) - len(filter_pass['kept'])))

    logging.warning('=' * 20)
    logging.warning('Filtered-out {:d} samples due to {:s}'
                    .format(len(filter_pass['{:s}'.format('animate_ref')]), 'animate_ref'))
    out_path = json_path[:-6] + '_{:s}.jsonl'.format('animate_ref')
    with open(out_path, 'w', encoding='utf8') as out_f:
        for s_iid, s_id in enumerate(filter_pass['{:s}'.format('animate_ref')].keys()):
            if s_iid > 0:
                out_f.write('\n')
            for i_id, item in enumerate(filter_pass['animate_ref'][s_id]):
                out_f.write(json.dumps(item))
                if i_id < 3:
                    out_f.write('\n')

    logging.warning('Identified {:d} samples that contain referents that are likely animals'
                    .format(len(filter_pass['animal_ref'])))
    out_path = json_path[:-6] + '_animal_ref.jsonl'
    with open(out_path, 'w', encoding='utf8') as out_f:
        for s_iid, s_id in enumerate(filter_pass['animal_ref'].keys()):
            if s_iid > 0:
                out_f.write('\n')
            for i_id, item in enumerate(filter_pass['animal_ref'][s_id]):
                out_f.write(json.dumps(item))
                if i_id < 3:
                    out_f.write('\n')

    logging.warning('Kept {:d} samples overall'.format(len(filter_pass['kept'])))
    out_path = json_path[:-6] + '_without_animate.jsonl'
    with open(out_path, 'w', encoding='utf8') as out_f:
        for s_iid, s_id in enumerate(filter_pass['kept'].keys()):
            if s_iid > 0:
                out_f.write('\n')
            for i_id, item in enumerate(filter_pass['kept'][s_id]):
                out_f.write(json.dumps(item))
                if i_id < 3:
                    out_f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the JSON file containing pre-filtered, annotated WinoGrande data')
    args = parser.parse_args()

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

    ctxs = [mx.gpu(0)]
    model, vocab, tokenizer = get_pretrained(ctxs, 'roberta-large-en-cased')
    scorer = MLMScorer(model, vocab, tokenizer, ctxs)

    annotate_samples(args.data_path)

