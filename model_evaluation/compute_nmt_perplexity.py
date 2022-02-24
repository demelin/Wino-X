import os
import torch
import fastBPE
import argparse

from fairseq import checkpoint_utils
from sacremoses import MosesTokenizer
from subword_nmt.apply_bpe import BPE, read_vocabulary

from eval_util import read_jsonl, segment_seq, prepare_inputs, compute_perplexity


def prepare_contra(json_file_path, out_dir_path):
    """ Pre-processes the contrastive translations corpus for NMT model evaluation. """

    # Read-in samples
    records = read_jsonl(json_file_path)

    # Decompose samples
    src_lines = list()
    tgt_lines = list()
    labels = list()

    for r in records:
        src_lines += [r['sentence']] * 2
        tgt_lines += [r['translation1'], r['translation2']]
        if r['answer'] == 1:
            labels += [1, 0]
        else:
            labels += [0, 1]

    # Write to files
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    src_out_path = '{:s}/src.txt'.format(out_dir_path)
    tgt_out_path = '{:s}/tgt.txt'.format(out_dir_path)
    labels_out_path = '{:s}/labels.txt'.format(out_dir_path)

    for lines, path in [(src_lines, src_out_path),
                        (tgt_lines, tgt_out_path),
                        (labels, labels_out_path)]:
        with open(path, 'w', encoding='utf8') as out_f:
            for line_id, line in enumerate(lines):
                out_f.write(str(line))
                if line_id < (len(lines) - 1):
                    out_f.write('\n')
        print('Wrote {:d} lines to {:s}'.format(len(lines), path))

    return src_out_path, tgt_out_path


def _score_sequence(src_sent, tgt_sent):
    """ Scores a source-target sentence pair using the translation model """
    src_sent_bpe, src_sent_ids = \
        segment_seq(src_sent, src_mt, src_bpe_model, model.encoder.dictionary, args.use_subword_nmt)
    tgt_sent_bpe, tgt_sent_ids = \
        segment_seq(tgt_sent, tgt_mt, tgt_bpe_model, model.decoder.dictionary, args.use_subword_nmt)
    sample = prepare_inputs(src_sent_ids, tgt_sent_ids, model, task, device)
    with torch.no_grad():
        _, _, log_output = criterion(model, sample, reduce=False)
    model_probabilities = log_output['target_probabilities']
    return compute_perplexity(model_probabilities)


def score_translations(src_path, tgt_path):
    """ Measures model perplexity for the specified source-translation pairs """

    # Read-in lines
    src_lines = list()
    with open(src_path, 'r', encoding='utf8') as src_f:
        for line in src_f:
            line = line.strip()
            if len(line) > 0:
                src_lines.append(line)

    tgt_lines = list()
    with open(tgt_path, 'r', encoding='utf8') as tgt_f:
        for line in tgt_f:
            line = line.strip()
            if len(line) > 0:
                tgt_lines.append(line)

    assert len(src_lines) == len(tgt_lines), 'Mismatched number of source and target lines!'

    # Score
    score_path = '/'.join(src_path.split('/')[:-1] + ['scores.txt'])
    all_scores = list()
    for line_id in range(len(src_lines)):
        with torch.no_grad():
            score = _score_sequence(src_lines[line_id], tgt_lines[line_id])
            all_scores.append(score)

        if (line_id + 1) % 1000 == 0:
            print('Scored {:d} translations'.format(line_id + 1))

    # Write to file
    with open(score_path, 'w', encoding='utf8') as sc_f:
        for score_id, score in enumerate(all_scores):
            sc_f.write(str(score))
            if score_id < len(all_scores) - 1:
                sc_f.write('\n')

    print('-' * 20)
    print('Wrote perplexity scores to {:s}'.format(score_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='path to the JSON file containing the contrastive samples')
    parser.add_argument('--src_path', type=str, default=None,
                        help='path to the text file containing evaluated source sentences')
    parser.add_argument('--tgt_path', type=str, default=None,
                        help='path to the text file containing target translations to be scored')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='path to the directory containing checkpoints')
    parser.add_argument('--checkpoint_names', type=str, required=True,
                        help='name(s) of checkpoints to use')
    parser.add_argument('--use_cpu', action='store_true',
                        help='whether to use the CPU for model passes'),
    parser.add_argument('--use_subword_nmt', action='store_true',
                        help='set to TRUE when evaluating the WMT14 en-fr model')
    parser.add_argument('--src_codes_path', type=str, required=True,
                        help='path to the files containing the source BPE-codes of the evaluated model')
    parser.add_argument('--tgt_codes_path', type=str, required=True,
                        help='path to the files containing the target BPE-codes of the evaluated model')
    parser.add_argument('--src_vocab_path', type=str, default=None,
                        help='path to the file containing model source vocabulary')
    parser.add_argument('--tgt_vocab_path', type=str, default=None,
                        help='path to the file containing model target vocabulary')
    parser.add_argument('--src_lang', type=str,
                        help='language code corresponding to the source language')
    parser.add_argument('--tgt_lang', type=str,
                        help='language code corresponding to the target language')
    args = parser.parse_args()

    # Generate NMT model inputs, if none are given
    if args.src_path is None or args.tgt_path is None:
        args.src_path, args.tgt_path = prepare_contra(args.json_file_path, args.out_dir)

    # Load model
    checkpoints_to_load = \
        [args.checkpoint_dir + '{:s}'.format(name) for name in args.checkpoint_names.split(':')]
    # if 'fair' in args.checkpoint_dir:
    #     arg_overrides = {'data': args.checkpoint_dir, 'cpu': args.use_cpu}
    # else:
    arg_overrides = {'cpu': args.use_cpu}
    models, saved_cfg, task = \
        checkpoint_utils.load_model_ensemble_and_task(checkpoints_to_load, arg_overrides=arg_overrides)
    model = models[0]

    device = 'cpu' if args.use_cpu else 'cuda:0'

    model.eval()
    if device != 'cpu':
        model.cuda()

    # Build criterion
    criterion = task.build_criterion('label_smoothed_cross_entropy_with_probs')
    criterion.eval()

    # Initialize tokenizers
    src_mt = MosesTokenizer(args.src_lang)
    tgt_mt = MosesTokenizer(args.tgt_lang)

    # Initialize BPE
    if args.use_subword_nmt:
        with open(args.src_vocab_path, 'r', encoding='utf8') as sv:
            src_vocab = read_vocabulary(sv, 50)
        with open(args.tgt_vocab_path, 'r', encoding='utf8') as tv:
            tgt_vocab = read_vocabulary(tv, 50)
        with open(args.src_codes_path, 'r', encoding='utf8') as sc:
            src_bpe_model = BPE(sc, vocab=src_vocab)
        with open(args.tgt_codes_path, 'r', encoding='utf8') as tc:
            tgt_bpe_model = BPE(tc, vocab=tgt_vocab)
    else:
        src_vocab, tgt_vocab = None, None
        src_bpe_model = fastBPE.fastBPE(args.src_codes_path, args.src_vocab_path)
        tgt_bpe_model = fastBPE.fastBPE(args.tgt_codes_path, args.tgt_vocab_path)

    score_translations(args.src_path, args.tgt_path)

