import argparse
from sacremoses import MosesTokenizer


def stitch_docs(doc1, doc2, out_file, tokenize_doc1, tokenize_doc2, remove_bpe):
    """ Pre-processes parallel text files for use with fast_align / awesome-align. """

    # Read
    with open(doc1, 'r', encoding='utf8') as f1:
        doc1_lines = f1.readlines()
    with open(doc2, 'r', encoding='utf8') as f2:
        doc2_lines = f2.readlines()

    doc1_lines = [line for line in doc1_lines if len(line.strip()) > 0]
    doc2_lines = [line for line in doc2_lines if len(line.strip()) > 0]

    # Optionally, tokenize docs
    if tokenize_doc1:
        doc1_lines = [mt.tokenize(line.strip(), return_str=True) for line in doc1_lines]
        d1_tok_out_path = '.'.join(doc1.split('.')[:-1]) + '.tok.' + doc1.split('.')[-1]
        with open(d1_tok_out_path, 'w', encoding='utf8') as out_f:
            for line_id, line in enumerate(doc1_lines):
                if line_id != 0:
                    out_f.write('\n')
                out_f.write(line)

    if tokenize_doc2:
        doc2_lines = [mt.tokenize(line.strip(), return_str=True) for line in doc2_lines]
        d2_tok_out_path = '.'.join(doc2.split('.')[:-1]) + '.tok.' + doc2.split('.')[-1]
        with open(d2_tok_out_path, 'w', encoding='utf8') as out_f:
            for line_id, line in enumerate(doc2_lines):
                if line_id != 0:
                    out_f.write('\n')
                out_f.write(line)

    # Prepare for aligning
    with open(out_file, 'w', encoding='utf8') as fo:
        for line_id, d1_line in enumerate(doc1_lines):
            d1_line = d1_line.strip()
            d2_line = doc2_lines[line_id].strip()
            if remove_bpe:
                d1_line = d1_line.replace('@@ ', '')
                d2_line = d2_line.replace('@@ ', '')
            if len(d1_line) > 0 and len(d2_line) > 0:
                fo.write('{:s} ||| {:s}\n'.format(d1_line, d2_line))
            # Report
            if line_id > 0 and line_id % 1000000 == 0:
                print('Joined {:d} lines'.format(line_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc1_path', type=str, required=True,
                        help='Path to the first document')
    parser.add_argument('--doc2_path', type=str, required=True,
                        help='Path to the second document')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Destination for the learned alignments')
    parser.add_argument('--tokenize_doc1', action='store_true',
                        help='Whether to tokenize doc1')
    parser.add_argument('--tokenize_doc2', action='store_true',
                        help='Whether to tokenize doc2')
    parser.add_argument('--tgt_lang', type=str, choices=['de', 'fr', 'ru'],
                        help='Code corresponding to the target language of the translations')
    parser.add_argument('--remove_bpe', action='store_true',
                        help='Whether to de-BPE lines')
    args = parser.parse_args()

    if args.tokenize_doc1 or args.tokenize_doc2:
        mt = MosesTokenizer(lang=args.tgt_lang)

    stitch_docs(args.doc1_path, args.doc2_path, args.out_path, args.tokenize_doc1, args.tokenize_doc2, args.remove_bpe)
