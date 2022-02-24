import os
import argparse
from util import read_jsonl


def construct_source(json_file, singular_only, use_inflected):
    """ Constructs source file to be given to an NMT model to obtain silver target translations. """

    records = read_jsonl(json_file)
    base_dir = '/'.join(json_file.split('/')[:-1])  # should be in its separate directory

    filled_path = base_dir + '/filled_sentences.en'
    ambiguous_path = base_dir + '/ambiguous_sentences.en'
    qids_path = base_dir + '/qids.lst'
    lines_written = 0

    with open(filled_path, 'w', encoding='utf8') as ff:
        with open(ambiguous_path, 'w', encoding='utf8') as af:
            with open(qids_path, 'w', encoding='utf8') as qf:
                for r_id, r in enumerate(records):
                    if singular_only:
                        if not (r['referent1_is_singular'] and r['referent2_is_singular']):
                            continue
                    if not use_inflected:
                        if r['referent1_is_inflected'] or r['referent2_is_inflected']:
                            continue
                    ff.write(r['filled_sentence'][0].upper() + r['filled_sentence'][1:]+'\n')
                    af.write(r['ambiguous_sentence'][0].upper() + r['ambiguous_sentence'][1:]+'\n')
                    qf.write(r['qID']+'\n')
                    lines_written += 1

    print('Wrote {:d} unambiguous sentences to {:s}'.format(lines_written, filled_path))
    print('Wrote {:d} ambiguous sentences to {:s}'.format(lines_written, ambiguous_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the JSON file containing the annotated WinoGrande data')
    parser.add_argument('--singular_only', action='store_true',
                        help='whether to only consider sentences with singular referents')
    parser.add_argument('--use_inflected_forms', action='store_true',
                        help='Whether to use samples containing inflected pronouns')
    args = parser.parse_args()

    construct_source(args.data_path, args.singular_only, args.use_inflected_forms)
