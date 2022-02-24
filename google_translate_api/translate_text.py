# Adopted from https://cloud.google.com/translate/docs/basic/translating-text#translate_translate_text-python

import argparse

from google.cloud import translate_v2 as translate


def translate_text(src_path, out_path, tgt_lang):
    """ Translates text into the target language. """

    # Initialize the translation client
    translate_client = translate.Client.from_service_account_json(
        './trial-translate-8cad7c67320d.json')

    # Read in source text
    lines = list()
    with open(src_path, 'r', encoding='utf8') as src_f:
        for line in src_f:
            line = line.strip()
            if len(line) > 0:
                lines.append(line.replace('&apos;', "'").replace('@-@', '-'))

    # Obtain translations
    results = list()
    for line_id, line in enumerate(lines):
        results.append(translate_client.translate(line, target_language=tgt_lang))
        if (line_id + 1) % 100 == 0:
            print('Translated {:d} sentences'.format(line_id + 1))

    # Do some reporting
    for r in results[:10]:
        print('Source: {}'.format(r['input']))
        print('Translation: {}'.format(r['translatedText']))

    # Save to file
    with open(out_path, 'w', encoding='utf8') as out_f:
        for r_id, r in enumerate(results):
            out_f.write(r['translatedText'])
            if r_id < len(results) - 1:
                out_f.write('\n')
    print('Wrote {:d} translated lines to {:s}'.format(len(results), out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True,
                        help='path to the text file containing source sentences to be translated')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the destination of the translated source sentences')
    parser.add_argument('--tgt_lang', type=str, required=True, choices=['de', 'fr', 'ru'],
                        help='Code corresponding to the target language of the translations')
    args = parser.parse_args()

    translate_text(args.src_path, args.out_path, args.tgt_lang)
