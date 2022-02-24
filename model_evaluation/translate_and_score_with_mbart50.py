import os
import torch
import string
import argparse
import sacrebleu

from transformers import (MBart50TokenizerFast,
                          MBartForConditionalGeneration)

MBART_TABLE = {'en': 'en_XX',
               'de': 'de_DE',
               'fr': 'fr_XX',
               'ru': 'ru_RU'}


def translate_and_score(src_path, tgt_path, out_dir, tgt_lang):
    """ Translates a test set with MBART50 and scores it against a reference. """

    # Read-in lines
    with open(src_path, 'r', encoding='utf8') as src_f:
        src_lines = src_f.readlines()
    with open(tgt_path, 'r', encoding='utf8') as tgt_f:
        tgt_lines = tgt_f.readlines()

    print('{:d} source lines | {:d} target lines'.format(len(src_lines), len(tgt_lines)))

    # Translate source lines
    translations = list()
    print('Translating ...')
    for line_id, line in enumerate(src_lines):
        if line_id > 0 and line_id % 1000 == 0:
            print('Translated {:d} lines'.format(line_id))
        encoded_src = tokenizer(line.strip(), return_tensors='pt')
        input_ids = encoded_src['input_ids'].to(device)
        mask = encoded_src['attention_mask'].to(device)
        generated_tokens = model.generate(input_ids=input_ids, attention_mask=mask, num_beams=5, early_stopping=True,
                                          no_repeat_ngram_size=3,
                                          forced_bos_token_id=tokenizer.lang_code_to_id[MBART_TABLE[tgt_lang]])
        translations += tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        print(translations)

    # Report SacreBLEU
    bleu = sacrebleu.corpus_bleu(translations, [[l.strip() for l in tgt_lines]])
    print('-' * 20)
    print('SacreBLEU results:')
    print(bleu.score)

    # Write translations to file
    out_path = '{:s}/{:s}_translations.{:s}'.format(out_dir, tgt_path.split('/')[-1], tgt_lang)
    with open(out_path, 'w', encoding='utf8') as out_f:
        for line_id, line in enumerate(translations):
            if line_id > 0:
                out_f.write('\n')
            out_f.write(line)
    print('Wrote translations to {:s}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True,
                        help='path to the source file of a bitext')
    parser.add_argument('--tgt_path', type=str, default=None,
                        help='path to the target file of a bitext')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Whether to use the CPU')
    parser.add_argument('--tgt_lang', type=str, choices=['de', 'fr', 'ru'],
                        help='Target language of the challenge set; ONLY AFFECTS THE ANSWERS, as the context and '
                             'question are always presented in English')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Instantiate model
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-one-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-one-to-many-mmt',
                                                     src_lang=MBART_TABLE['en'], tgt_lang=MBART_TABLE[args.tgt_lang])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    translate_and_score(args.src_path, args.tgt_path, args.out_dir, args.tgt_lang)
