import argparse


def unpack(xlf_path, src_lang, tgt_lang):
    """ Parses the weirdly formatted RAPID corpus into an actually useful format """

    # Parse
    src_lines = list()
    tgt_lines = list()
    with open(xlf_path, 'r', encoding='utf8') as in_f:
        for line in in_f:
            if line.startswith('<source') or line.startswith('<target'):
                line = line.replace('<', '|').replace('>', '|')
                segments = [seg.strip() for seg in line.split('|') if len(seg.strip()) > 0]
                if "lang=\"{:s}\"".format(src_lang) in segments[0]:
                    src_lines.append(segments[1])
                if "lang=\"{:s}\"".format(tgt_lang) in segments[0]:
                    tgt_lines.append(segments[1])

    assert len(src_lines) == len(tgt_lines), 'Source and target sentences differ in number!'

    # Write
    src_path = '.'.join(xlf_path.split('.')[:-1]) + '.{:s}'.format(src_lang)
    tgt_path = '.'.join(xlf_path.split('.')[:-1]) + '.{:s}'.format(tgt_lang)

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
    parser.add_argument('--xlf_path', type=str, required=True,
                        help='path to the .xlf file containing the corpus')
    parser.add_argument('--src_lang', type=str, required=True,
                        help='source language ID')
    parser.add_argument('--tgt_lang', type=str, required=True,
                        help='target language ID')
    args = parser.parse_args()

    unpack(args.xlf_path, args.src_lang, args.tgt_lang)
