import os
import math
import json
import torch
import logging
import argparse

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader,
                              RandomSampler,
                              SequentialSampler,
                              TensorDataset)

from transformers import (AutoTokenizer,
                          BertConfig,
                          BertForMaskedLM,
                          BertTokenizer,
                          RobertaConfig,
                          RobertaForMaskedLM,
                          MBartConfig,
                          MBartTokenizer,
                          MBart50TokenizerFast,
                          MBartForConditionalGeneration,
                          AdamW,
                          get_linear_schedule_with_warmup)

from util import (WinoXMTProcessor, WinoXMLMPronProcessor, WinoXMLMWordsProcessor, WinoFullProcessor,
                  compute_gen_metrics, convert_examples_to_features,
                  set_seed, _rotate_checkpoints, get_token_loss)

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, AutoTokenizer),
    'xlm-roberta': (RobertaConfig, RobertaForMaskedLM, AutoTokenizer),
    'mbart': (MBartConfig, MBartForConditionalGeneration, MBartTokenizer),
    'mbart50': (MBartConfig, MBartForConditionalGeneration, MBart50TokenizerFast),
}
TASKS = ['contra', 'mlm', 'mlm_nouns', 'winogrande_full']

MBART_TABLE = {'en': 'en_XX',
               'de': 'de_DE',
               'fr': 'fr_XX',
               'ru': 'ru_RU'}

TASK_DICT = {}
MAX_GEN_LENGTH = int(10000)


def train(args, model, tokenizer):
    """ Trains the model. """
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    # Set-up TensorBoard logger
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Set up data-serving pipeline
    train_dataset = load_and_cache_examples(args, tokenizer, split='train')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Estimate the number of update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Set-up weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Set-up optimizer and schedule (linear warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=math.floor(args.warmup_pct * t_total), num_training_steps=t_total)

    # Set-up mixed precision training
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Set-up multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Set-up distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Learning rate = {}'.format(args.learning_rate))
    logger.info('  Instantaneous batch size per GPU = %d', args.per_gpu_train_batch_size)
    logger.info('  Total train batch size (w. parallel, distributed & accumulation) = %d',
                args.train_batch_size * args.gradient_accumulation_steps *
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info('  Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', t_total)

    # Initialize tracking variables
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    curr_eval_loss = float('inf')
    best_eval_loss = float('inf')
    # curr_metric = 0.0
    # best_metric = 0.0
    stale_epochs = 0

    # Iterate through training data
    best_checkpoint_path = None
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    for epoch_count in train_iterator:
        epoch_iterator = \
            tqdm(train_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0], mininterval=10, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            # logger.info('Epoch: {} | Step: {}'.format(epoch_count, step))
            # Perform a single training step
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type in ['bert', 'roberta', 'xlm-roberta']:  # i.e. encoder-decoder models
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type == 'bert' else None,
                          'labels': batch[3]}

            else:
                # I.e. mBART, mBART50
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3][:, 1:].contiguous()}
                with tokenizer.as_target_tokenizer():
                    # Prepare decoder inputs and labels for enc-dec models
                    decoder_input_ids = batch[3][:, :-1].clone()  # shift
                    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
                    inputs['decoder_input_ids'] = decoder_input_ids.contiguous()

            outputs = model(**inputs)
            loss = outputs[0]

            # Compute distributed loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Compute mixed-precision loss
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Back-propagate
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        # Log metrics and evaluate model performance on the validation set
        mean_epoch_loss = (tr_loss - logging_loss) / len(epoch_iterator)
        logging.info('\n' + '*' * 10)
        logging.info('Mean epoch training loss: {:.4f}'.format(mean_epoch_loss))
        logging.info('*' * 10 + '\n')

        # Log metrics and evaluate model performance on the validation set
        if args.local_rank in [-1, 0]:
            # Only evaluate when single GPU otherwise metrics may not average well
            if args.local_rank == -1 and args.evaluate_during_training:
                results, curr_eval_loss = evaluate(args, model, tokenizer, 'dev', global_step)
                for key, value in results.items():
                    tb_writer.add_scalar('eval_{}'.format(key), value[-1], global_step)
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('train_loss', mean_epoch_loss, global_step)
            tb_writer.add_scalar('eval_loss', curr_eval_loss, global_step)
            # tb_writer.add_scalar('metric', curr_metric, global_step)
            logging_loss = tr_loss

        # Save model checkpoint
        if args.local_rank in [-1, 0]:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(args.task_name, epoch_count))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed / parallel training
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args_{}.bin'.format(args.task_name)))
            logger.info('Saving model checkpoint to %s', output_dir)
            # Delete old checkpoints to maintain a low memory footprint
            _rotate_checkpoints(args, 'checkpoint-{}'.format(args.task_name), use_mtime=False)

        # Check whether to stop training early (can use loss or challenge accuracy as criterion)
        if curr_eval_loss < best_eval_loss:
            best_eval_loss = curr_eval_loss
            stale_epochs = 0
            best_checkpoint_path = \
                os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(args.task_name, global_step))
        else:
            stale_epochs += 1
            logging.info(
                '!!! Development loss has not improved this epoch. Stale epochs: {:d} !!!'.format(stale_epochs))

        if stale_epochs >= args.patience:
            logging.info(
                '\n***** STOPPING TRAINING EARLY AFTER {:d} STALE VALIDATION STEPS *****\n'.format(stale_epochs))
            break

        if epoch_count > args.num_train_epochs > 0:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_checkpoint_path


def evaluate(args, model, tokenizer, split, step):
    """ Evaluates models on dev / test sets. """
    model.eval()
    results = dict()

    # Reference previous evaluation results
    if os.path.exists(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split))):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split)), 'r') as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    # Set up data-serving pipeline
    eval_dataset = load_and_cache_examples(args, tokenizer, split=split)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info('***** Running evaluation on the validation / test set *****')
    logger.info('  Num examples = %d', len(eval_dataset))
    logger.info('  Batch size = %d', args.eval_batch_size)
    batch_losses = list()
    eval_loss = 0.0
    micro_loss, macro_loss = 0.0, 0.0
    num_batches, num_tokens = 0, 0
    # Perform a single evaluation step
    for batch in tqdm(eval_dataloader, desc='Evaluating', mininterval=10, ncols=100):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.model_type in ['bert', 'roberta', 'xlm-roberta']:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type == 'bert' else None,
                          'labels': batch[3]}
            else:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3][:, 1:].contiguous()}
                with tokenizer.as_target_tokenizer():
                    # Prepare decoder inputs and labels for enc-dec models
                    decoder_input_ids = batch[3][:, :-1].clone()  # shift
                    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
                    inputs['decoder_input_ids'] = decoder_input_ids.contiguous()

            outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()
        batch_losses.append(tmp_eval_loss.mean().item())

        # Obtain per-token loss for perplexity computation
        batch_loss = get_token_loss(args, logits, batch[3], batch[4])
        macro_loss += batch_loss.mean().item()
        micro_loss += batch_loss.sum().item()
        num_batches += 1
        num_tokens += batch_loss.view(-1).shape[0]

    # Compute and update evaluation metric values
    macro_perplexity = torch.exp(torch.tensor(macro_loss / num_batches)).item()
    micro_perplexity = torch.exp(torch.tensor(micro_loss / num_tokens)).item()
    curr_result = {'macro_perplexity': macro_perplexity,
                   'micro_perplexity': micro_perplexity}

    if len(results.keys()) == 0:
        for k, v in curr_result.items():
            results[k] = [v]
    else:
        for k, v in curr_result.items():
            results[k].append(v)

    # Log metrics
    output_eval_file = os.path.join(args.output_dir, 'results_{}_{}.txt'.format(args.task_name, split))
    with open(output_eval_file, 'a') as writer:
        logger.info('***** Eval results *****')
        writer.write('STEP: {:s}\n'.format(str(step)))
        for key in sorted(curr_result.keys()):
            logger.info('  %s = %s', key, str(curr_result[key]))
            writer.write('%s = %s\n' % (key, str(curr_result[key])))

    # Maintain a single metrics file
    if os.path.exists(args.output_dir):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split)), 'w') as f:
            f.write(json.dumps(results))
        f.close()

    # Report mean dev loss
    mean_eval_loss = eval_loss / len(eval_dataloader)
    logging.info('\n' + '*' * 10)
    logging.info('Mean development loss: {:.4f}'.format(mean_eval_loss))
    logging.info('*' * 10 + '\n')

    return results, mean_eval_loss


def test_gen(args, model, tokenizer, split, step):
    """ Evaluates generative models on the test set. """

    assert args.model_type in ['bert', 'roberta', 'xlm-roberta'], \
        'Specified model-type is incompatible with generation tests'

    results = dict()
    # Reference previous evaluation results
    if os.path.exists(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split))):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split)), 'r') as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    # Set up data-serving pipeline
    test_dataset = load_and_cache_examples(args, tokenizer, split=split)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    # Test!
    logger.info('***** Testing generation on the test set *****')
    logger.info('  Num examples = %d', len(test_dataset))
    logger.info('  Batch size = %d', test_batch_size)
    generation_inputs = list()
    generation_targets = list()
    generated_sequences = list()

    # Iterate through the test corpus
    model.eval()
    for batch in tqdm(test_dataloader, desc='Testing', mininterval=10, ncols=100):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            # Modify inputs (assumes a batch size of 1)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            max_gen_length = args.max_gen_length
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     min_length=5,
                                     max_length=max_gen_length,
                                     temperature=args.temperature,
                                     top_k=args.k if args.k > 0 else None,
                                     top_p=args.p if args.p > 0 else None,
                                     num_beams=args.num_beams if args.num_beams > 0 else None,
                                     do_sample=args.do_sample,
                                     early_stopping=True,
                                     no_repeat_ngram_size=3,
                                     decoder_start_token_id=model.config.decoder_start_token_id)

            # Remove the batch dimension when returning multiple sequences
            if len(outputs.shape) > 2:
                outputs.squeeze_()

            # Convert model predictions to text sequences
            input_ids = inputs['input_ids'].tolist()
            target_ids = inputs['labels'].tolist()
            # Post-process model predictions and prediction targets
            for generated_sequence_idx, generated_sequence in enumerate(outputs):
                generated_sequence = generated_sequence.tolist()

                # Prepare predictions
                generated_sequence = generated_sequence[: generated_sequence.index(tokenizer.eos_token_id)]
                decoded_generated_sequence = \
                    tokenizer.decode(generated_sequence[1:], clean_up_tokenization_spaces=True)
                generated_sequences.append(decoded_generated_sequence)

                # Prepare generation inputs
                generation_input = input_ids[generated_sequence_idx]
                generation_input = generation_input[: generation_input.index(tokenizer.pad_token_id)]
                generation_inputs.append(tokenizer.decode(generation_input, clean_up_tokenization_spaces=True))

                # Prepare generation targets
                gen_target = target_ids[generated_sequence_idx]
                gen_target = gen_target[: gen_target.index(tokenizer.eos_token_id)]
                decoded_generation_target = tokenizer.decode(gen_target[1:], clean_up_tokenization_spaces=True)
                generation_targets.append(decoded_generation_target)

    # Report sample generation results
    logging.info('***** Example generations *****')
    for s_id, gen_input in enumerate(generation_inputs):
        if s_id >= 10:
            break
        logging.info('  Inputs: {:s}'.format(gen_input))
        logging.info('  Reference: {:s}'.format(generation_targets[s_id]))
        logging.info('  Prediction: {:s}'.format(generated_sequences[s_id]))

    # Compute and update evaluation metric values
    curr_result = compute_gen_metrics(generated_sequences, generation_targets)
    if len(results.keys()) == 0:
        for k, v in curr_result.items():
            results[k] = [v]
    else:
        for k, v in curr_result.items():
            results[k].append(v)

    # Log metrics
    output_eval_file = \
        os.path.join(args.output_dir, 'generation_test_results_{}_{}_{}.txt'.format(args.task_name, split, step))
    with open(output_eval_file, 'w') as writer:
        logger.info('***** Test results *****')
        writer.write('STEP: {:s}\n'.format(str(step)))
        for key in sorted(curr_result.keys()):
            logger.info('  %s = %s', key, str(curr_result[key]))
            writer.write('%s = %s\n' % (key, str(curr_result[key])))

    # Log predictions
    output_pred_file = \
        os.path.join(args.output_dir, 'generation_test_predictions_{}_{}_{}.lst'.format(args.task_name, split, step))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions *****')
        for gsi, gs in enumerate(generated_sequences):
            writer.write(json.dumps({'prefix': generation_inputs[gsi],
                                     'target': generation_targets[gsi],
                                     'prediction': gs}) + '\n')

    # Maintain a single metrics file
    if os.path.exists(args.output_dir):
        with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
            f.write(json.dumps(results))
        f.close()
    return results


def load_and_cache_examples(args, tokenizer, split):
    """ Prepares the dataset splits for use with the model. """
    # Set processor
    if args.task_name == 'winogrande_full':
        processor = WinoFullProcessor(args)
    elif args.task_name == 'contra':
        processor = WinoXMTProcessor(args)
    elif args.task_name == 'mlm':
        processor = WinoXMLMPronProcessor(args)
    else:
        processor = WinoXMLMWordsProcessor(args)

    logger.info('Creating features from dataset file at %s', args.data_dir)
    if split == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif split == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    elif split == 'test':
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception('split value should be in [train, dev, test]')

    # Generate features
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    if pad_token_id is None:
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]

    features = convert_examples_to_features(examples,
                                            args.model_type,
                                            tokenizer,
                                            max_seq_len=args.max_seq_length,
                                            pad_token=pad_token_id,
                                            sequence_a_segment_id=0,
                                            pad_token_segment_id=0,
                                            mask_padding_with_zero=True)

    # Make feature tensors
    all_src_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
    all_src_masks = torch.tensor([f.src_mask for f in features], dtype=torch.long)
    all_src_segments = torch.tensor([f.src_segments for f in features], dtype=torch.long)
    all_tgt_ids = torch.tensor([f.tgt_ids for f in features], dtype=torch.long)
    all_tgt_masks = torch.tensor([f.tgt_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_src_ids, all_src_masks, all_src_segments, all_tgt_ids, all_tgt_masks)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='The input data dir. Should contain the .tsv files (or other data files) for the task.')
    parser.add_argument('--model_type', default=None, type=str, required=True, choices=list(MODEL_CLASSES.keys()),
                        help='Model type selected in the list: ' + ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help='Path to pre-trained model')
    parser.add_argument('--task_name', default=None, type=str, required=True, choices=TASKS,
                        help='The name of the task to train selected in the list: ' + ', '.join(TASKS))
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='The root output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--src_lang', default=None, type=str, required=True,
                        help='Source language ID')
    parser.add_argument('--tgt_lang', default=None, type=str, required=True,
                        help='Target language ID')

    ## Generation parameters
    parser.add_argument('--max_gen_length', default=60, type=int,
                        help='The maximum length of the sequence to be generated.')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='The value used to module the next token probabilities.')
    parser.add_argument('--k', default=0, type=int,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--p', default=0, type=float,
                        help='If set to float < 1, only the most probable tokens with probabilities that add up to '
                             'top_p or higher are kept for generation.')
    parser.add_argument('--num_beams', default=0, type=int, required=False, help='beams for beam search')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to generate predictions via sampling; if off, decoding is done greedily.')
    parser.add_argument('--patience', default=-1, type=int, required=False,
                        help='Number of epochs without dev-set loss improvement to ignore before '
                             'stopping the training.')
    parser.add_argument('--run_on_cpu', action='store_true',
                        help='Whether to run the experiment on the CPU.')

    ## Other parameters
    parser.add_argument('--config_name', default='', type=str,
                        help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', default='', type=str,
                        help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--cache_dir', default='', type=str,
                        help='The cache directory where do you want to store the pre-trained models downloaded from s3')
    parser.add_argument('--max_seq_length', default=-1, type=int,
                        help='The maximum total input sequence length after tokenization. Sequences longer '
                             'than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_prediction', action='store_true',
                        help='Whether to run prediction on the test set. (Training will not be executed.)')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='Do evaluation during training at each logging step.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Set this flag if you are using an uncased model.')
    parser.add_argument('--run_on_test', action='store_true',
                        help='Evaluate model on the test split.')
    parser.add_argument('--data_cache_dir', default=None, type=str,
                        help='The root directory for caching features.')
    parser.add_argument('--pretrained_dir', default=None, type=str,
                        help='The directory containing the checkpoint of a pretrained model to be fine-tuned.')
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Maximum number of checkpoints to keep')

    parser.add_argument('--per_gpu_train_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight deay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument('--num_train_epochs', default=3.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', default=0, type=int,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--warmup_pct', default=None, type=float,
                        help='Linear warmup over warmup_pct * total_steps.')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log every X updates steps.')
    parser.add_argument('--save_epochs', type=int, default=1,
                        help='Save checkpoint every X updates steps.')
    parser.add_argument('--eval_all_checkpoints', action='store_true',
                        help='Evaluate all checkpoints starting with the same prefix as model_name ending and ending '
                             'with step number')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the content of the output directory')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help='For fp16: Apex AMP optimization level selected in [\'O0\', \'O1\', \'O2\', and \'O3\'].'
                             'See details at https://nvidia.github.io/apex/amp.html')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='For distributed training: local_rank')
    parser.add_argument('--server_ip', type=str, default='', help='For distant debugging.')
    parser.add_argument('--server_port', type=str, default='', help='For distant debugging.')
    args = parser.parse_args()

    # Create output / cache directory paths
    args.data_cache_dir = os.path.join(args.data_cache_dir, args.task_name, args.model_type,
                                       '{}_{}'.format(args.src_lang, args.tgt_lang))
    # Check if directories need to be created
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.data_cache_dir):
        os.makedirs(args.data_cache_dir)

    # Setup distant debugging, if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logging.info('Waiting for debugger attach ...')
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes / GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    if args.run_on_cpu:
        args.device = 'cpu'

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning('Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Set processor
    if args.task_name == 'winogrande_full':
        processor = WinoFullProcessor(args)
    elif args.task_name == 'contra':
        processor = WinoXMTProcessor(args)
    elif args.task_name == 'mlm':
        processor = WinoXMLMPronProcessor(args)
    else:
        processor = WinoXMLMWordsProcessor(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Instantiate model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model, tokenizer = None, None

    if args.do_train:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path, finetuning_task=args.task_name)
        if args.model_type == 'mbart50':
            src_lang_id = MBART_TABLE[args.src_lang]
            tgt_lang_id = MBART_TABLE[args.tgt_lang]
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else
                                                        args.model_name_or_path, src_lang=src_lang_id,
                                                        tgt_lang=tgt_lang_id)
        else:
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else
                                                        args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        if args.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        processor.tokenizer = tokenizer
        model.to(args.device)

    logger.info('Training / evaluation parameters %s', args)
    # Training
    final_checkpoint_name = args.output_dir
    if args.do_train:
        global_step, tr_loss, final_checkpoint_name = train(args, model, tokenizer)
        logger.info(' global_step = %s, average loss = %s', global_step, tr_loss)

    # Evaluate on dev
    checkpoints = [final_checkpoint_name]
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint = checkpoints[0]
        if not args.do_train:
            checkpoint = args.model_name_or_path
        global_step = int(checkpoint.split('-')[-1])
        model = model_class.from_pretrained(checkpoint)
        if args.model_type == 'mbart50':
            src_lang_id = MBART_TABLE[args.src_lang]
            tgt_lang_id = MBART_TABLE[args.tgt_lang]
            tokenizer = tokenizer_class.from_pretrained(checkpoint, src_lang=src_lang_id, tgt_lang=tgt_lang_id)
        else:
            tokenizer = tokenizer_class.from_pretrained(checkpoint)  # hack
        model.to(args.device)
        dev_split = 'dev' if args.do_train else 'test'
        results, loss = evaluate(args, model, tokenizer, dev_split, global_step)
        logger.info('-' * 20)
        logger.info('Global step: {:d}'.format(global_step))
        logger.info('Loss: {:.3f}'.format(loss))
        logger.info('Results:')
        logger.info(results)

    # Evaluate on test
    if args.do_prediction and args.local_rank in [-1, 0]:
        checkpoint = checkpoints[0]
        if not args.do_train:
            checkpoint = args.model_name_or_path
        global_step = int(checkpoint.split('-')[-1])
        model = model_class.from_pretrained(checkpoint)
        if args.model_type == 'mbart50':
            src_lang_id = MBART_TABLE[args.src_lang]
            tgt_lang_id = MBART_TABLE[args.tgt_lang]
            tokenizer = tokenizer_class.from_pretrained(checkpoint, src_lang=src_lang_id, tgt_lang=tgt_lang_id)
        else:
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
        model.to(args.device)
        logger.info('Prediction on the test set (note: Training will not be executed.) ')
        if 'gen' not in args.task_name:
            evaluate(args, model, tokenizer, 'test', global_step)
        else:
            test_gen(args, model, tokenizer, 'test', global_step)

    logger.info('***** Experiment finished *****')


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logging.info(exception)
        raise Exception(exception)

