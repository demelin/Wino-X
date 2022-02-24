import argparse

from sacremoses import MosesTokenizer


def post_process(in_path, out_path, tgt_lang):
    """ Fixes apostrophes and tokenizes translations obtained from Google Translate. """

    # Read-in translations
    lines = list()
    with open(in_path, 'r', encoding='utf8') as in_f:
        # Fix apostrophes
        for line in in_f:
            line = line.strip()
            if tgt_lang in ['fr', 'ru']:
                line = line.replace("&#39;", "'")
            else:
                # German is a special case
                line = line.replace("Rocky &#39;s", "Rockys").replace("Kathy &#39;s", "Kathys")  # hack
                tokens = line.split()
                fixed_tokens = list()
                for tok in tokens:
                    if "&#39;" in tok:
                        if tok.split("&#39;")[-1] == "s":
                            tok = tok.replace("&#39;", "")
                        else:
                            tok = tok.replace("&#39;", "'")
                    fixed_tokens.append(tok)
                line = ' '.join(fixed_tokens).replace("  ", " ")

            # Moses-tokenize
            line = mt.tokenize(line, return_str=True)
            lines.append(line)

    # Write
    with open(out_path, 'w', encoding='utf8') as out_f:
        for line_id, line in enumerate(lines):
            out_f.write(line)
            if line_id < len(lines) - 1:
                out_f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True,
                        help='path to the text file containing "raw" translations')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the destination of the post-processed translations')
    parser.add_argument('--tgt_lang', type=str, required=True, choices=['de', 'fr', 'ru'],
                        help='Code corresponding to the target language of the translations')
    args = parser.parse_args()

    mt = MosesTokenizer(lang=args.tgt_lang)

    post_process(args.in_path, args.out_path, args.tgt_lang)
