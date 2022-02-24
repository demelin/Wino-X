import json
import argparse

from util import read_jsonl


def do_filter(winogrande_path, test_path, dev_path, out_path):
    """ Removes test and dev samples from the full WinoGrande corpus """

    # Read
    full_samples = read_jsonl(winogrande_path)
    test_samples = read_jsonl(test_path)
    dev_samples = read_jsonl(dev_path)
    filter_ids = set([s['qID'].split('-')[0].strip() for s in test_samples] +
                     [s['qID'].split('-')[0].strip() for s in dev_samples])

    # Filter
    keep = list()
    num_dropped = 0
    for s in full_samples:
        if s['qID'].split('-')[0].strip() not in filter_ids:
            s['context_en'] = s['sentence'][:]
            s['option1_en'] = s['option1'][:]
            s['option2_en'] = s['option2'][:]
            del s['sentence']
            del s['option1']
            del s['option2']
            keep.append(s)
        else:
            num_dropped += 1

    # Write
    with open(out_path, 'w', encoding='utf8') as out_f:
        for s_id, s in enumerate(keep):
            if s_id > 0:
                out_f.write('\n')
            out_f.write(json.dumps(s, ensure_ascii=False))
    print('-' * 5)
    print('Wrote {:d} samples to {:s}'.format(len(keep), out_path))
    print('Filtered out {:d} samples'.format(num_dropped))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--winogrande_path', type=str, required=True,
                        help='path to the WinoGrande file')
    parser.add_argument('--dev_path', type=str, required=True,
                        help='path to the dev file')
    parser.add_argument('--test_path', type=str, required=True,
                        help='path to the test file')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the output file')

    args = parser.parse_args()

    do_filter(args.winogrande_path, args.test_path, args.dev_path, args.out_path)
