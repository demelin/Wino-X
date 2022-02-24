import argparse


def unpack(tsv_path):
    """ Parses the weirdly formatted RAPID corpus into an actually useful format """

    # Identify src and tgt language
    src_lang, tgt_lang = tsv_path.split('.')[-2].split('-')

    # Parse
    src_lines = list()
    tgt_lines = list()
    with open(tsv_path, 'r', encoding='utf8') as in_f:
        for line in in_f:
            segments = [seg.strip() for seg in line.split('\t') if len(seg.strip()) > 0]
            try:
                src_line = segments[0]
                tgt_line = segments[1]
            except IndexError:
                continue
            src_lines.append(src_line)
            tgt_lines.append(tgt_line)

    assert len(src_lines) == len(tgt_lines), 'Source and target sentences differ in number!'

    # Write
    src_path = '.'.join(tsv_path.split('.')[:-1]) + '.{:s}'.format(src_lang)
    tgt_path = '.'.join(tsv_path.split('.')[:-1]) + '.{:s}'.format(tgt_lang)

    print('-' * 20)
    with open(src_path, 'w', encoding='utf8') as src_f:
        for line_id, line in enumerate(src_lines):
            src_f.write(line)
            if line_id < len(src_lines) - 1:
                src_f.write('\n')
    print('Wrote {:d} source lines to {:s}'.format(line_id + 1, src_path))

    with open(tgt_path, 'w', encoding='utf8') as tgt_f:
        for line_id, line in enumerate(tgt_lines):
            tgt_f.write(line)
            if line_id < len(tgt_lines) - 1:
                tgt_f.write('\n')
    print('Wrote {:d} target lines to {:s}'.format(line_id + 1, tgt_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str, required=True,
                        help='path to the .tsf file containing the corpus')
    args = parser.parse_args()

    unpack(args.tsv_path)
