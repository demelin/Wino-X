import langid
import argparse


def enforce_lang(tok_path_src, bpe_path_src, tok_path_tgt, bpe_path_tgt, src_lang, tgt_lang):
    """ Filters out lines from the parallel dataset that are not identified to belong to the correct language """

    # Read
    src_tok_lines = list()
    tgt_tok_lines = list()

    src_bpe_lines = list()
    tgt_bpe_lines = list()

    for path, lines in [(tok_path_src, src_tok_lines),
                        (bpe_path_src, src_bpe_lines),
                        (tok_path_tgt, tgt_tok_lines),
                        (bpe_path_tgt, tgt_bpe_lines)]:

        print('Reading in {:s} ...'.format(path))
        with open(path, 'r', encoding='utf8') as in_f:
            for line in in_f:
                lines.append(line)
        print('Read-in {:d} lines from {:s}'.format(len(lines), path))

    # Filter
    print('-' * 20)
    print('Filtering lines ...')
    filtered_src_bpe_lines = list()
    filtered_tgt_bpe_lines = list()

    for line_id, src_line in enumerate(src_tok_lines):
        src_line = src_line.strip()
        tgt_line = tgt_tok_lines[line_id].strip()
        if len(src_line) <= 0 or len(tgt_line) <= 0:
            continue
        src_line_lang = langid.classify(src_line)[0]
        if src_line_lang != src_lang:
            continue
        tgt_line_lang = langid.classify(tgt_line)[0]
        if tgt_line_lang != tgt_lang:
            continue
        filtered_src_bpe_lines.append(src_bpe_lines[line_id])
        filtered_tgt_bpe_lines.append(tgt_bpe_lines[line_id])

    assert len(filtered_src_bpe_lines) == len(filtered_tgt_bpe_lines), \
        'Unequal number of filtered source and target languages!'
    print('Kept {:d} (source) / {:d} (target) BPE\'d lines after filtering!'.format(len(filtered_src_bpe_lines),
                                                                                    len(filtered_tgt_bpe_lines)))

    # Write
    print('-' * 20)
    src_out_path = '.'.join(bpe_path_src.split('.')[:-1]) + '.langid.{:s}'.format(src_lang)
    tgt_out_path = '.'.join(bpe_path_tgt.split('.')[:-1]) + '.langid.{:s}'.format(tgt_lang)
    for path, lines in [(src_out_path, filtered_src_bpe_lines), (tgt_out_path, filtered_tgt_bpe_lines)]:
        print('Writing to {:s} ...'.format(path))
        with open(path, 'w', encoding='utf8') as out_f:
            for line in lines:
                out_f.write(line)
        print('Wrote {:d} lines to {:s}'.format(len(lines), path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tok_path_src', type=str, required=True,
                        help='path to the source tokenized file')
    parser.add_argument('--bpe_path_src', type=str, required=True,
                        help='path to the source BPE\'d file')
    parser.add_argument('--tok_path_tgt', type=str, required=True,
                        help='path to the target tokenized file')
    parser.add_argument('--bpe_path_tgt', type=str, required=True,
                        help='path to the target BPE\'d file')
    parser.add_argument('--src_lang', type=str, required=True,
                        help='language code for the intended source language')
    parser.add_argument('--tgt_lang', type=str, required=True,
                        help='language code for the intended target language')
    args = parser.parse_args()

    langid.set_languages([args.src_lang, args.tgt_lang])
    enforce_lang(
        args.tok_path_src, args.bpe_path_src, args.tok_path_tgt, args.bpe_path_tgt, args.src_lang, args.tgt_lang)
