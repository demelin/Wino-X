import os
import json
import random
import argparse

from util import read_jsonl


def split_dataset(json_file_path, train_size, dev_size, test_size, out_dir, src_lang, tgt_lang):
    """ Splits the challenge set into train / dev / test """

    assert train_size % 2 == 0 and dev_size % 2 == 0, 'train_size and dev_size must be divisible by 2'

    # Read
    samples = read_jsonl(json_file_path)
    pairs = list()
    curr_pair = list()
    for s_id, s in enumerate(samples):
        curr_pair.append(s)
        if len(curr_pair) == 2:
            pairs.append(curr_pair)
            curr_pair = list()

    # Report
    print('-' * 10)
    print('Total # samples in the challenge set: {:d}'.format(len(samples)))
    print('Total # pairs in the challenge set: {:d}'.format(len(pairs)))

    # Assign
    if args.shuffle:
        random.seed(42)
        random.shuffle(pairs)

    # For iterative few-shot fine-tuning; keep test and dev constant, but increase train
    if test_size > 0:
        test_pairs = pairs[: (test_size // 2)]
        dev_pairs = pairs[(test_size // 2): (test_size // 2) + (dev_size // 2)]
        train_pairs = pairs[(test_size // 2) + (dev_size // 2): (test_size // 2) + (dev_size // 2) + (train_size // 2)]

    else:
        train_pairs = pairs[: (train_size // 2)]
        dev_pairs = pairs[(train_size // 2): (train_size // 2) + (dev_size // 2)]
        test_pairs = pairs[(train_size // 2) + (dev_size // 2):]

    train_split, dev_split, test_split = list(), list(), list()
    for pairs, split, key in [(train_pairs, train_split, 'train'),
                              (dev_pairs, dev_split, 'dev'),
                              (test_pairs, test_split, 'test')]:

        src_lines, tgt_lines = list(), list()
        for pair in pairs:
            # Keep full samples for LM fine-tuning
            split += pair
            # Generate plain-text files for MT fine-tuning
            if 'contra' in json_file_path:
                for s in pair:
                    src_lines.append(s['sentence'])
                    tgt_lines.append(s['translation1'] if int(s['answer']) == 1 else s['translation2'])

        # Write to files
        out_path = '{:s}/{:s}.jsonl'.format(out_dir, key, src_lang, tgt_lang)
        with open(out_path, 'w', encoding='utf8') as out_f:
            for s_id, s in enumerate(split):
                if s_id > 0:
                    out_f.write('\n')
                out_f.write(json.dumps(s,ensure_ascii=False))
        print('-' * 5)
        print('Wrote {:d} {:s} samples to {:s}'.format(len(split), key, out_path))

        if len(src_lines) > 0:
            nmt_dir = '{:s}/{:s}'.format(out_dir, 'nmt')
            if not os.path.exists(nmt_dir):
                os.makedirs(nmt_dir)

            key_dir = '{:s}/{:s}'.format(nmt_dir, key)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir)
            src_lines_path = '{:s}/{:s}.{:s}'.format(key_dir, key, src_lang)
            with open(src_lines_path, 'w', encoding='utf8') as src_f:
                for l_id, l in enumerate(src_lines):
                    if l_id > 0:
                        src_f.write('\n')
                    src_f.write(l)
            print('Wrote {:d} {:s} source lines to {:s}'.format(len(src_lines), key, src_lines_path))

            tgt_lines_path = '{:s}/{:s}.{:s}'.format(key_dir, key, tgt_lang)
            with open(tgt_lines_path, 'w', encoding='utf8') as tgt_f:
                for l_id, l in enumerate(tgt_lines):
                    if l_id > 0:
                        tgt_f.write('\n')
                    tgt_f.write(l)
            print('Wrote {:d} {:s} target lines to {:s}'.format(len(tgt_lines), key, tgt_lines_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='path to the JSON file containing challenge set samples')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--train_size', type=int, default=500,
                        help='number of samples to include in the training set')
    parser.add_argument('--dev_size', type=int, default=200,
                        help='number of samples to include in the development set')
    parser.add_argument('--test_size', type=int, default=1000,
                        help='number of samples to include in the development set')
    parser.add_argument('--src_lang', type=str, required=True, choices=['en'],
                        help='code corresponding to the target language of the translations')
    parser.add_argument('--tgt_lang', type=str, required=True, choices=['de', 'fr', 'ru'],
                        help='code corresponding to the target language of the translations')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data')
    args = parser.parse_args()

    # Create output directory, if necessary
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    split_dataset(args.json_file_path, args.train_size, args.dev_size, args.test_size, args.out_dir, args.src_lang,
                  args.tgt_lang)
