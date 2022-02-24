# -*- coding: utf-8 -*-

import sys
from sacremoses import MosesDetokenizer


def _filter_translations(trans_file, out_file):
    """ Extracts target-side translations from the fairseq-interactive output. """
    with open(trans_file, 'r', encoding='utf8') as inf:
        with open(out_file, 'w', encoding='utf8') as ouf:
            for line in inf:
                # Example: H-781   -0.35308536887168884    Oh, mein Gott! Es ist ein großer Tag!
                if line.startswith('H'):
                    trans_sent = line.strip().split('\t')[-1]
                    trans_sent = md.detokenize(trans_sent.split())
                    ouf.write(trans_sent + '\n')

    print('Done')


if __name__ == '__main__':
    md = MosesDetokenizer(lang=sys.argv[3])
    _filter_translations(sys.argv[1], sys.argv[2])
