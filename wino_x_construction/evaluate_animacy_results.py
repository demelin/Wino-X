import json
import argparse

from util import read_jsonl


def evaluate_samples(json_path):
    """ Displays possibly animate WinoGrande samples for manual evaluation. """

    # Read
    records = read_jsonl(json_path)

    # Pair
    paired_samples = dict()
    for s_iid, s in enumerate(records):
        if 'option1_is_animate' in s.keys():
            continue
        s_id = s['qID'].split('-')[0]
        if not paired_samples.get(s_id, None):
            paired_samples[s_id] = [s]
        else:
            paired_samples[s_id].append(s)

    # Evaluate
    not_animate = dict()
    for s_iid, s_id in enumerate(paired_samples.keys()):
        pair = paired_samples[s_id]
        print('=' * 10)
        print(pair[0]['sentence'])
        print(pair[0]['option1'])
        print(pair[0]['option2'])
        retrieve = ''
        while retrieve not in ['1', '2']:
            retrieve = input('Retrieve sentence? ')
        if retrieve == '2':
            not_animate[s_id] = pair

    # Write
    print('=' * 20)
    out_path = json_path[:-6] + '_{:s}.jsonl'.format('corrected_nonperson')
    with open(out_path, 'w', encoding='utf8') as out_f:
        for s_iid, s_id in enumerate(not_animate.keys()):
            if s_iid > 0:
                out_f.write('\n')
            for i_id, item in enumerate(not_animate[s_id]):
                out_f.write(json.dumps(item))
                if i_id < 3:
                    out_f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the JSON file containing pre-filtered, annotated WinoGrande data')
    args = parser.parse_args()

    evaluate_samples(args.data_path)
