import json
import random
import stanza
import string
import argparse
import unidecode

from Bio import pairwise2
from sacremoses import MosesDetokenizer

from util import read_jsonl

GER_PRONOUNS = {'Sing|Masc|Nom': 'er',
                'Sing|Fem|Nom': 'sie',
                'Sing|Neut|Nom': 'es',

                'Plur|Masc|Nom': 'sie',
                'Plur|Fem|Nom': 'sie',
                'Plur|Neut|Nom': 'sie'}

FR_PRONOUNS = {'Sing|Masc|Nom': 'il',
               'Sing|Fem|Nom': 'elle',

               'Plur|Masc|Nom': 'ils',
               'Plur|Fem|Nom': 'elles'}

RU_PRONOUNS = {'Sing|Masc|Nom': 'oн',
               'Sing|Fem|Nom': 'oнa',
               'Sing|Neut|Nom': 'oнo',

               'Plur|Masc|Nom': 'oни',
               'Plur|Fem|Nom': 'oни',
               'Plur|Neut|Nom': 'oни'}


def _map_between_tokens(ws_tokens, other_tokens):
    """ Learns a map between two sets of tokens corresponding to the same sentence. """

    print('+' * 20)
    print(ws_tokens)
    print(other_tokens)

    # Generate a mapping between whitespace tokens and SpaCy tokens
    other_to_ws_map = dict()
    ws_to_other_map = dict()
    ws_loc = 0
    ws_tok = ws_tokens[ws_loc]

    for other_loc, other_tok in enumerate(other_tokens):
        while True:

            # print(ws_loc, ws_tok)

            if other_tok == ws_tok or other_tok in ws_tok:
                # Terminate
                if ws_loc >= len(ws_tokens):
                    break
                # Extend maps
                if not ws_to_other_map.get(ws_loc, None):
                    ws_to_other_map[ws_loc] = list()
                ws_to_other_map[ws_loc].append(other_loc)
                if not other_to_ws_map.get(other_loc, None):
                    other_to_ws_map[other_loc] = list()
                other_to_ws_map[other_loc].append(ws_loc)
                # Move pointer
                if other_tok == ws_tok:
                    ws_loc += 1
                    if ws_loc < len(ws_tokens):
                        ws_tok = ws_tokens[ws_loc]
                else:
                    ws_tok = ws_tok[len(other_tok):]
                break
            else:
                ws_loc += 1

    # Assert full coverage of whitespace and SpaCy token sequences by the mapping
    ws_covered = sorted(list(ws_to_other_map.keys()))
    other_covered = sorted(list(set(list([val for val_list in ws_to_other_map.values() for val in val_list]))))
    assert ws_covered == [n for n in range(len(ws_tokens))], \
        'WS-OTHER mapping does not cover all whitespace tokens: {}; number of tokens: {}' \
            .format(ws_covered, len(ws_tokens))
    assert other_covered == [n for n in range(len(other_tokens))], \
        'WS-OTHER mapping does not cover all SpaCy tokens: {}; number of tokens: {}' \
            .format(other_covered, len(other_tokens))

    return ws_to_other_map, other_to_ws_map


def _insert_target_pronoun(translations, annotations, pronoun_list, tgt_lang, is_true):
    """ Inserts the appropriate target language pronoun in the provided translations. """

    # For each src-tgt-alignment pairing in translations_table
    # 1. Lookup corresponding JSON sample [X]
    # 2. Identify referent to be replaced with pronoun [X]
    # 3. Check alignment of referent position with target [X]
    # 4. Obtain morphological features for target sentence and aligned target word [X]
    # 5. Replace target word with appropriate pronoun [X]
    # 6. Detokenize target sentences [X]

    # Map WS-tokens to the Moses tokens of the translation source
    src_tokens = \
        translations['src'].strip().replace('&apos;', '\'').replace('@-@', '-').replace("&amp; quot ;", "\"").split()
    ws_tokens = annotations['filled_sentence'].strip().split()
    ws_to_other_map, _ = _map_between_tokens(ws_tokens, src_tokens)
    # Process translation
    tgt_sentence = \
        translations['tgt'].strip().replace('&apos;', '\'').replace('@-@', '-').replace("&amp; quot ;", "\"")
    tgt_tokens = tgt_sentence.split()
    if 'The' in tgt_tokens or 'the' in tgt_tokens:
        return 'bad_translation'

    print('-' * 10)
    print(ws_tokens)
    print(src_tokens)
    print(ws_to_other_map)
    print(annotations['ws_gap_ids'])
    print(tgt_tokens)

    # Look up locations of the gap-filler
    filler_src_loc = list()
    for ws_id in annotations['ws_gap_ids']:
        src_ids = ws_to_other_map.get(ws_id, None)
        if src_ids is not None:
            filler_src_loc += [sid for sid in src_ids if src_tokens[sid] not in string.punctuation]

    # Look up locations of the referent ('alt' refers to the incorrect alternative co-referent)
    referent_ws_loc = annotations['ws_option1_position'] if annotations['answer'] == '1' else \
        annotations['ws_option2_position']
    alt_referent_ws_loc = annotations['ws_option2_position'] if annotations['answer'] == '1' else \
        annotations['ws_option1_position']
    if not is_true:
        temp = referent_ws_loc
        referent_ws_loc = alt_referent_ws_loc
        alt_referent_ws_loc = temp
    # Map positions to the Moses-tokenized text
    referent_src_loc = list()
    for ws_id in referent_ws_loc:
        src_ids = ws_to_other_map.get(ws_id, None)
        if src_ids is not None:
            referent_src_loc += [sid for sid in src_ids if src_tokens[sid] not in string.punctuation]
    alt_referent_src_loc = list()
    for ws_id in alt_referent_ws_loc:
        src_ids = ws_to_other_map.get(ws_id, None)
        if src_ids is not None:
            alt_referent_src_loc += [sid for sid in src_ids if src_tokens[sid] not in string.punctuation]

    # This improves recall, as aligner often connects English articles with Russian nouns (bit of a hack)
    if tgt_lang == 'ru':
        if filler_src_loc[0] > 0 and src_tokens[filler_src_loc[0] - 1].lower() in ['a', 'an', 'the']:
            filler_src_loc = [filler_src_loc[0] - 1] + filler_src_loc
        if referent_src_loc[0] > 0 and src_tokens[referent_src_loc[0] - 1].lower() in ['a', 'an', 'the']:
            referent_src_loc = [referent_src_loc[0] - 1] + referent_src_loc
        if alt_referent_src_loc[0] > 0 and src_tokens[alt_referent_src_loc[0] - 1].lower() in ['a', 'an', 'the']:
            alt_referent_src_loc = [alt_referent_src_loc[0] - 1] + alt_referent_src_loc

    print('-' * 10)
    print(annotations['ws_gap_ids'], filler_src_loc)
    print(referent_ws_loc, referent_src_loc)
    print(alt_referent_ws_loc, alt_referent_src_loc)

    # Check which target words the gap-filler and its referents are aligned with
    alignment_table = dict()
    for align_pair in translations['alignment'].split():
        src_loc, tgt_loc = align_pair.split('-')
        src_loc = int(src_loc)
        tgt_loc = int(tgt_loc)
        if alignment_table.get(src_loc, None) is None:
            alignment_table[src_loc] = [tgt_loc]
        else:
            alignment_table[src_loc].append(tgt_loc)
    # Look up locations of the gap-filler's translation
    filler_tgt_loc = list()
    for fsl in filler_src_loc:
        tgt_ids = alignment_table.get(fsl, None)
        if tgt_ids is not None:
            filler_tgt_loc += tgt_ids
    # Look up locations of the true referent's translation
    referent_tgt_loc = list()
    for rsl in referent_src_loc:
        tgt_ids = alignment_table.get(rsl, None)
        if tgt_ids is not None:
            referent_tgt_loc += tgt_ids
    # Look up locations of the false referent's translation
    alt_referent_tgt_loc = list()
    for rsl in alt_referent_src_loc:
        tgt_ids = alignment_table.get(rsl, None)
        if tgt_ids is not None:
            alt_referent_tgt_loc += tgt_ids

    # Remove duplicates / referent overlap
    filler_tgt_loc = list(set(filler_tgt_loc))
    referent_tgt_loc = list(set(referent_tgt_loc))
    alt_referent_tgt_loc = list(set(alt_referent_tgt_loc))
    referent_overlap = set(referent_tgt_loc) & set(alt_referent_tgt_loc)
    if len(referent_overlap) > 0:
        if len(referent_tgt_loc) > len(alt_referent_tgt_loc):
            referent_tgt_loc = [l for l in referent_tgt_loc if l not in referent_overlap]
        else:
            alt_referent_tgt_loc = [l for l in alt_referent_tgt_loc if l not in referent_overlap]

    print('/-' * 10)
    print(alignment_table)
    print(filler_tgt_loc)
    print(referent_tgt_loc)
    print(alt_referent_tgt_loc)

    # Handle instances where no alignment could be found
    if len(filler_tgt_loc) == 0 or len(referent_tgt_loc) == 0 or len(alt_referent_tgt_loc) == 0:
        return 'no_alignment'

    # Parse target translation with Stanza
    filler_phrase = ' '.join([tgt_tokens[ftl] for ftl in filler_tgt_loc])
    parsed_tgt_init = nlp(tgt_sentence)
    # Account for multi-sentence translations
    parsed_tgt = list()
    for sent_id in range(len(parsed_tgt_init.sentences)):
        for word in parsed_tgt_init.sentences[sent_id].words:
            parsed_tgt.append(word)
    original_parsed_target = parsed_tgt

    if len(tgt_tokens) != len(parsed_tgt):
        # Post-process Stanza parse
        # 1. Find best alignment between token sequences using the Needleman-Wunsch algorithm
        # 2. Exclude positions from the Stanza parse that correspond to gaps in the alignment
        stanza_tokens = [entry.text for entry in parsed_tgt]
        alignment = pairwise2.align.globalxx(tgt_tokens, stanza_tokens, gap_char=['<GAP>'])[0]
        aligned_tgt_tokens = alignment.seqA
        aligned_stanza_tokens = alignment.seqB

        print('-' * 10)
        print(tgt_tokens)
        print(stanza_tokens)
        print(aligned_tgt_tokens)
        print(aligned_stanza_tokens)

        filtered_parsed_tgt = list()
        num_gaps = 0
        for i, tok in enumerate(aligned_tgt_tokens):
            if tok != '<GAP>':
                if aligned_stanza_tokens[i] != '<GAP>':
                    filtered_parsed_tgt.append(parsed_tgt[i - num_gaps])
                else:
                    filtered_parsed_tgt.append(None)
                    num_gaps += 1
        parsed_tgt = filtered_parsed_tgt

    # Identify the replacement pronoun
    noun_parse, filler_noun_id, filler_noun_feats, filler_pos = None, None, None, None
    for ftl in filler_tgt_loc:
        parse = parsed_tgt[ftl]
        if parse is None:
            continue
        if parse.upos in ['NOUN', 'PROPN', 'PRON']:  # filler is sometimes translated as pronoun
            noun_parse = parse
            filler_pos = parse.upos
    try:
        noun_feats = noun_parse.feats.split('|')
    except AttributeError:

        print('-' * 20)
        print('NOUN PARSE')
        print(noun_parse)

        return 'bad_morphology'

    print('-' * 20)
    print('FILLER:')
    print(noun_parse)

    filler_noun_feats = {tpl.split('=')[0]: tpl.split('=')[1] for tpl in noun_feats}  # use final ref noun
    # Skip not easily substitutable cases
    if filler_noun_feats is None or (tgt_lang != 'fr' and filler_noun_feats.get('Case', None) != 'Nom'):
        return 'bad_morphology'

    # Parse post-filler gap and ensure that, if verbs are present, the initial verb agrees in number with the filler
    if filler_tgt_loc[-1] + 1 < len(parsed_tgt):
        parsed_suffix = parsed_tgt[filler_tgt_loc[-1] + 1:]
        for w in parsed_suffix:
            if w is not None:
                try:
                    if w.upos in ['VERB', 'AUX']:
                        verb_feats = {feat.split('=')[0]: feat.split('=')[1] for feat in w.feats.split('|')}
                        verb_num = verb_feats.get('Number', None)
                        filler_num = filler_noun_feats.get('Number', None)
                        if verb_num is not None and filler_num is not None and verb_num != filler_num:
                            return 'bad_morphology'
                        else:
                            break
                except AttributeError:
                    continue


    noun_parse, referent_det_feats, referent_noun_id, referent_noun_feats, referent_lemmas, referent_tokens = \
        None, None, None, None, list(), list()
    # Check the true co-referent's features
    for rtl in referent_tgt_loc:
        parse = parsed_tgt[rtl]
        if parse is None:
            continue
        if parse.upos in ['NOUN', 'PROPN']:
            noun_parse = parse
        referent_lemmas.append(parse.lemma.split('|')[0])  # true co-referent phrase
        referent_tokens.append(parse.text)
    try:
        noun_feats = noun_parse.feats.split('|')
        referent_noun_id = noun_parse.id
    except AttributeError:
        return 'bad_morphology'

    print('-' * 20)
    print('TRUE REF:')
    print(noun_parse)

    referent_noun_feats = {tpl.split('=')[0]: tpl.split('=')[1] for tpl in noun_feats}

    if tgt_lang == 'de':
        # Find determiner of true co-referent
        for j in range(len(original_parsed_target)):
            if original_parsed_target[j].id == referent_noun_id:
                break
            if original_parsed_target[j] is not None and original_parsed_target[j].head == referent_noun_id and \
                    original_parsed_target[j].upos == 'DET':

                print('-' * 20)
                print('TRUE REF DET:')
                print(original_parsed_target[j])

                referent_det_feats = \
                    {tpl.split('=')[0]: tpl.split('=')[1] for tpl in original_parsed_target[j].feats.split('|')}

    alt_referent_lemmas, alt_referent_tokens = list(), list()
    # Check the false co-referent's features
    for rtl in alt_referent_tgt_loc:
        parse = parsed_tgt[rtl]
        if parse is None:
            continue
        if parse.upos in ['NOUN', 'PROPN']:
            noun_parse = parse
        alt_referent_lemmas.append(parse.lemma.split('|')[0])  # true co-referent phrase
        alt_referent_tokens.append(parse.text)
    try:
        noun_feats = noun_parse.feats.split('|')
    except AttributeError:
        return 'bad_morphology'

    print('-' * 20)
    print('FALSE REF:')
    print(noun_parse)

    alt_referent_noun_feats = {tpl.split('=')[0]: tpl.split('=')[1] for tpl in noun_feats}

    # Check for determiner before the gap-filler
    filler_det = ''
    if tgt_lang != 'ru':
        try:
            if parsed_tgt[filler_tgt_loc[0] - 1].upos == 'DET':
                filler_tgt_loc = [filler_tgt_loc[0] - 1] + filler_tgt_loc
                filler_det = parsed_tgt[filler_tgt_loc[0]].text
            else:
                print('Pre-gap determiner not found.')
                if filler_pos != 'PRON':
                    return 'bad_translation'
        except AttributeError:

            print('===>>> LINE 349')

            return 'bad_translation'

    # Replace gap-filler with the appropriate target pronoun

    print('=' * 20)
    print(filler_noun_feats)
    print(referent_det_feats)

    referent_feats = referent_det_feats if tgt_lang == 'de' else referent_noun_feats
    case = 'Nom'  # filler pronoun is always expected to be in the nominative case, due to prior filtering
    # Enforce referent agreement
    if referent_feats is not None:
        referent_num = referent_feats.get('Number', None)
        referent_gender = referent_feats.get('Gender', None)
    else:
        referent_num = None
        referent_gender = None
    filler_num = filler_noun_feats.get('Number', None)
    filler_gender = filler_noun_feats.get('Gender', None)
    if filler_num != referent_num or (tgt_lang in ['fr', 'ru'] and filler_gender != referent_gender):

        print('===>>> LINE 369')
        print(original_parsed_target)

        return 'bad_translation'

    num = referent_num
    gender = referent_gender
    if num is None or gender is None:
        return 'bad_morphology'
    # Case is expected to be Nominative
    pronoun_key = '{:s}|{:s}|{:s}'.format(num, gender, case)
    tgt_pronoun = pronoun_list[pronoun_key]

    print('-' * 10)
    print(translations['tgt'].strip().split())
    print(filler_tgt_loc)

    tgt_context = list()
    for tok_id, tok in enumerate(translations['tgt'].strip().split()):
        if tok_id not in filler_tgt_loc:
            # Exceptions
            if len(tgt_context) > 0 and tok == '@-@' and tgt_context[-1] == '_':
                continue
            if len(tgt_context) - 1 > 0 and tgt_context[-1] == '@-@' and tgt_context[-2] == '_':
                continue
            tgt_context.append(tok)
        elif tok_id == filler_tgt_loc[0]:
            tgt_context.append('_')
        else:
            continue

    # Detokenize target and fix apostrophes
    tgt_with_pronoun = detokenizer.detokenize(tgt_context)
    pronoun_loc = [tok_id for tok_id, tok in enumerate(tgt_with_pronoun.split()) if '_' in tok][0]
    detokenized_tgt = tgt_with_pronoun.replace('_', tgt_pronoun)
    # Account for French elision:
    if tgt_lang == 'fr':
        detok_tokens = detokenized_tgt.split()
        if unidecode.unidecode(detok_tokens[pronoun_loc - 1][-1]) in 'aeiou':
            detok_tokens[pronoun_loc - 1] = detok_tokens[pronoun_loc - 1][:-1] + "'"
            detokenized_tgt = ' '.join(detok_tokens)

    return detokenized_tgt, tgt_with_pronoun, pronoun_key, tgt_pronoun, ' '.join(referent_lemmas), \
        ' '.join(alt_referent_lemmas), ' '.join(referent_tokens), ' '.join(alt_referent_tokens), \
        referent_noun_feats, alt_referent_noun_feats, filler_phrase, filler_noun_feats, filler_det, filler_pos


def build_corpora(json_src_path, src_path, tgt_path, qid_path, alignments_path, tgt_lang, out_path,
                  exclude_multi_sentence_samples):
    """ Constructs Wino-X datasets for the evaluation of NMT and X-LM models. """

    # Read
    samples = read_jsonl(json_src_path)
    samples_table = dict()
    id2samples = dict()
    for s in samples:
        samples_table[s['qID']] = s

        qid = s['qID'].split('-')[0] if s['qID'].count('-') == 1 else '-'.join(s['qID'].split('-')[:-1])
        sid = int(s['qID'].split('-')[-1])

        # Fix capitalization
        s['sentence'] = s['sentence'][0].upper() + s['sentence'][1:].replace("’", "'")
        s['filled_sentence'] = s['filled_sentence'][0].upper() + s['filled_sentence'][1:].replace("’", "'")
        s['ambiguous_sentence'] = s['ambiguous_sentence'][0].upper() + s['ambiguous_sentence'][1:].replace("’", "'")

        if id2samples.get(qid, None) is None:
            id2samples[qid] = {1: {'true_ref': None, 'false_ref': None},
                               2: {'true_ref': None, 'false_ref': None}}
        if s['referent_is_true']:
            id2samples[qid][sid]['true_ref'] = s
        else:
            id2samples[qid][sid]['false_ref'] = s

    translated_qid = list()
    translated_src = list()
    translated_tgt = list()
    alignments = list()

    for path, a_list in [(qid_path, translated_qid),
                         (src_path, translated_src),
                         (tgt_path, translated_tgt),
                         (alignments_path, alignments)]:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    a_list.append(line)
    assert len(translated_qid) == len(translated_src) == len(translated_tgt) == len(alignments), \
        'Length mismatch between the translated components: qid {}, src {}, tgt {}, alignments {}'.format(
            len(translated_qid), len(translated_src), len(translated_tgt), len(alignments))

    # Consolidate sentences, their translations, and alignments
    # qID -> 1|2 -> true|false src -> true|false tgt, alignment
    translations_table = dict()
    counter = 0
    for i_id, i in enumerate(translated_qid):
        counter += 1
        qid = i.split('-')[0] if i.count('-') == 1 else '-'.join(i.split('-')[:-1])
        sid = int(i.split('-')[-1])
        ref_key = 'true_ref' if counter <= 2 else 'false_ref'
        if translations_table.get(qid, None) is None:
            translations_table[qid] = {1: {'true_ref': {'src': None, 'tgt': None, 'alignment': None},
                                           'false_ref': {'src': None, 'tgt': None, 'alignment': None}},
                                       2: {'true_ref': {'src': None, 'tgt': None, 'alignment': None},
                                           'false_ref': {'src': None, 'tgt': None, 'alignment': None}}}
        translations_table[qid][sid][ref_key]['src'] = translated_src[i_id].replace("’", "'")
        translations_table[qid][sid][ref_key]['tgt'] = translated_tgt[i_id]
        translations_table[qid][sid][ref_key]['alignment'] = alignments[i_id]
        counter = 0 if counter >= 4 else counter

    # Initialize challenge set containers
    contra_data = list()
    mlm_data = list()
    mlm_nouns_data = list()
    qa_data = list()

    samples_with_no_alignment = set()
    samples_with_bad_morphology = set()
    samples_with_bad_translations = set()
    samples_with_matching_pronouns = set()

    if tgt_lang == 'de':
        pronoun_list = GER_PRONOUNS
    elif tgt_lang == 'fr':
        pronoun_list = FR_PRONOUNS
    else:
        pronoun_list = RU_PRONOUNS

    def _skip_sample(_qid, exclude_pronoun_match=False):
        """ Checks whether the sample has been rejected or not. """
        _qid = _qid.split('-')[0] if _qid.count('-') == 1 else '-'.join(_qid.split('-')[:-1])

        if exclude_pronoun_match:
            return _qid in samples_with_no_alignment or _qid in samples_with_bad_morphology or \
                   _qid in samples_with_bad_translations
        else:
            return _qid in samples_with_no_alignment or _qid in samples_with_bad_morphology or \
                _qid in samples_with_bad_translations or _qid in samples_with_matching_pronouns

    def _report_stats(is_final=False):
        """ Helper function for tracking dataset creation progress. """
        print('=' * 20)
        if is_final:
            print('Finished dataset construction!')
            print('-' * 10)
        print('Processed {:d} WinoGrande samples'.format(qid_id + 1))
        print('-' * 10)
        print('Created {:d} CONTRASTIVE samples'.format(
            len([ex for ex in contra_data if not _skip_sample(ex['qID'])])))
        print('Examples:')
        for _s in contra_data[-10:]:
            print(_s)
        print('-' * 10)
        print('Created {:d} MLM samples'.format(
            len([ex for ex in mlm_data if not _skip_sample(ex['qID'])])))
        print('Examples:')
        for _s in mlm_data[-10:]:
            print(_s)
        print('-' * 10)
        print('Created {:d} QA samples'.format(
            len([ex for ex in qa_data if not _skip_sample(ex['qID'])])))
        print('Examples:')
        for _s in qa_data[-10:]:
            print(_s)
        print('=' * 10)

    # Construct challenge set samples
    for qid_id, qid in enumerate(translations_table.keys()):

        # Skip certain samples
        # Multi-sentence parents
        if exclude_multi_sentence_samples and (samples_table[qid + '-1']['is_multi_sentence'] is True or
                                               samples_table[qid + '-2']['is_multi_sentence'] is True):
            continue

        # trigger-phrases containing 'it'
        mod_punctuation = string.punctuation.replace('_', '')
        s1_tokens = [tok.strip(mod_punctuation) for tok in samples_table[qid + '-1']['sentence'].lower().split()]
        s2_tokens = [tok.strip(mod_punctuation) for tok in samples_table[qid + '-2']['sentence'].lower().split()]
        if 'it' in s1_tokens[s1_tokens.index('_') + 1:] or 'it' in s1_tokens[s2_tokens.index('_') + 1:]:
            continue

        pair_translations = translations_table[qid]

        print('=' * 10)
        print(pair_translations)

        pair_annotations = id2samples[qid]
        for sid in translations_table[qid].keys():
            # Insert target pronoun
            true_ref_translations = pair_translations[sid]['true_ref']
            true_ref_annotations = pair_annotations[sid]['true_ref']
            true_referent_out = \
                _insert_target_pronoun(true_ref_translations, true_ref_annotations, pronoun_list, tgt_lang, True)
            # Repeat for false co-referent
            false_ref_translations = pair_translations[sid]['false_ref']
            false_ref_annotations = pair_annotations[sid]['false_ref']
            false_referent_out = \
                _insert_target_pronoun(false_ref_translations, false_ref_annotations, pronoun_list, tgt_lang, False)

            print('=' * 20)
            print(true_referent_out)
            print(false_referent_out)

            if type(true_referent_out) == str:
                if true_referent_out == 'no_alignment':
                    samples_with_no_alignment.add(qid)
                if true_referent_out == 'bad_morphology':
                    samples_with_bad_morphology.add(qid)
                if true_referent_out == 'bad_translation':
                    samples_with_bad_translations.add(qid)
                continue
            if type(false_referent_out) == str:
                if false_referent_out == 'no_alignment':
                    samples_with_no_alignment.add(qid)
                if false_referent_out == 'bad_morphology':
                    samples_with_bad_morphology.add(qid)
                if false_referent_out == 'bad_translation':
                    samples_with_bad_translations.add(qid)
                continue

            true_ref_tgt_with_pronoun, true_ref_tgt_context, true_pronoun_key, true_ref_tgt_pronoun, \
                true_ref_lemmas_true, false_ref_lemmas_true, true_ref_tokens_true, false_ref_tokens_true, \
                true_ref_feats, false_ref_feats, true_filler, true_filler_feats, true_filler_det, true_filler_pos = \
                true_referent_out
            false_ref_tgt_with_pronoun, false_ref_tgt_context, false_pronoun_key, false_ref_tgt_pronoun,\
                true_ref_lemmas_false, false_ref_lemmas_false, true_ref_tokens_false, false_ref_tokens_false, \
                _, _, false_filler, false_filler_feats, false_filler_det, false_filler_pos = false_referent_out

            print('=' * 20)
            print(true_ref_tgt_with_pronoun)
            print(false_ref_tgt_with_pronoun)
            print(true_ref_lemmas_true, true_ref_lemmas_false)
            print(false_ref_lemmas_true, false_ref_lemmas_false)

            # Ensure that both translations have identical co-reference candidates, based on lemmas
            if true_ref_lemmas_true != false_ref_lemmas_false or true_ref_lemmas_false != false_ref_lemmas_true:
                # samples_with_bad_translations.add('{:s}-{:d}'.format(qid, sid))
                samples_with_bad_translations.add(qid)

                print('=' * 20)
                print('REFERENT MISMATCH!')

                continue

            # Construct dataset samples
            # 1: Contrastive translations ('contra')
            # 2: Cross-lingual masked language modeling with pronouns ('mlm', close to translation objective)
            # 3: Cross-lingual masked language modeling with nouns ('mlm_nouns')
            true_id = random.choice([1, 2])
            if ((true_filler_feats.get('Case', None) == 'Nom' and false_filler_feats.get('Case', None) == 'Nom') or
                    tgt_lang == 'fr') and not (true_filler_pos == 'PRON' or false_filler_pos == 'PRON') and \
                    (true_filler_det is not None and false_filler_det is not None):
                # Somewhat weird setup: Provide context in source language and provide answers in target language
                # (is this an established cross-lingual QA paradigm?)

                # Assign source language fillers (for QA, MLM)
                src_sample = samples_table['{:s}-{:d}'.format(qid, sid)]
                true_src_filler = src_sample['option1'] if src_sample['answer'] == '1' else src_sample['option2']
                false_src_filler = src_sample['option2'] if src_sample['answer'] == '1' else src_sample['option1']
                src_filler1 = true_src_filler if true_id == 1 else false_src_filler
                src_filler2 = false_src_filler if true_id == 1 else true_src_filler

                # Assign target language fillers
                noun1, det1 = (true_filler, true_filler_det) if true_id == 1 else (false_filler, false_filler_det)
                noun2, det2 = (false_filler, false_filler_det) if true_id == 1 else (true_filler, true_filler_det)
                noun1 = noun1.split()[-1].strip()
                noun2 = noun2.split()[-1].strip()
                # Equalize
                if tgt_lang in ['fr', 'ru']:
                    noun1 = noun1.lower() if not noun1.isupper() else noun1
                    noun2 = noun2.lower() if not noun2.isupper() else noun2
                else:
                    noun1 = noun1.capitalize() if not noun1.isupper() else noun1
                    noun2 = noun2.capitalize() if not noun2.isupper() else noun2

                # Assign referents
                true_trans_ref1 = true_ref_tokens_true if true_id == 1 else false_ref_tokens_true
                true_trans_ref2 = false_ref_tokens_true if true_id == 1 else true_ref_tokens_true

                # Construct question phrase
                parsed_src = nlp_src(true_ref_annotations['sentence']).sentences[0].words
                src_loc = 0
                for src_loc in range(len(parsed_src)):
                    if parsed_src[src_loc].text == '_':
                        break
                suffix = parsed_src[src_loc + 1:] if len(parsed_src) > (src_loc + 1) else None
                # Modify suffix, if needed
                if suffix is not None:
                    suffix_pos = [w.upos for w in suffix]
                    suffix_words = [w.text for w in suffix]
                    if suffix_pos[0] not in ['VERB', 'AUX']:
                        for sfx_loc in range(len(suffix_pos)):
                            if suffix_pos[sfx_loc] in ['VERB', 'AUX']:
                                suffix_words = suffix_words[sfx_loc:]
                                break
                        if suffix_pos[0] not in ['VERB', 'AUX']:
                            suffix_words = ['is'] + suffix_words

                    prompt = 'What {:s}?'.format(' '.join(suffix_words).replace(' ,', ',').replace(' .', ''))

                    if noun1 != noun2:
                        # Non-English question phrases are non-trivial to construct
                        new_sample_qa = {'qID': '{:s}-{:d}'.format(qid, sid),
                                         'sentence'.format(tgt_lang): true_ref_annotations['ambiguous_sentence'],
                                         'question': prompt,
                                         'option1_en': 'the {:s}'.format(src_filler1),
                                         'option2_en': 'the {:s}'.format(src_filler2),
                                         'option1_{:s}'.format(tgt_lang): '{:s} {:s}'.format(det1, noun1),
                                         'option2_{:s}'.format(tgt_lang): '{:s} {:s}'.format(det2, noun2),
                                         'answer': true_id}
                        qa_data.append(new_sample_qa)

                if (true_ref_tgt_pronoun == false_ref_tgt_pronoun or tgt_lang == 'de') and noun1 != noun2:
                    # MLM with word-fillers has similar prerequisites as QA
                    new_sample_mlm_nouns = {'qID': '{:s}-{:d}'.format(qid, sid),
                                            'sentence': true_ref_annotations['ambiguous_sentence'],
                                            'context_en': src_sample['sentence'].replace('the _', '_'),
                                            'context_{:s}'.format(tgt_lang): true_ref_tgt_context,
                                            'option1_en': 'the {:s}'.format(src_filler1),
                                            'option2_en': 'the {:s}'.format(src_filler2),
                                            'option1_{:s}'.format(tgt_lang): '{:s} {:s}'.format(det1, noun1),
                                            'option2_{:s}'.format(tgt_lang): '{:s} {:s}'.format(det2, noun2),
                                            'answer': true_id,

                                            # Added for analysis
                                            'context_referent_of_option1_{:s}'.format(tgt_lang): true_trans_ref1,
                                            'context_referent_of_option2_{:s}'.format(tgt_lang): true_trans_ref2}
                    mlm_nouns_data.append(new_sample_mlm_nouns)

            # Ignore samples with identical pronouns in correct and incorrect translations
            if true_ref_tgt_pronoun != false_ref_tgt_pronoun:
                translation1 = true_ref_tgt_with_pronoun if true_id == 1 else false_ref_tgt_with_pronoun
                translation2 = false_ref_tgt_with_pronoun if true_id == 1 else true_ref_tgt_with_pronoun
                pronoun1 = true_ref_tgt_pronoun if true_id == 1 else false_ref_tgt_pronoun
                pronoun2 = false_ref_tgt_pronoun if true_id == 1 else true_ref_tgt_pronoun

                # Assign source language fillers (for QA, MLM)
                src_sample = samples_table['{:s}-{:d}'.format(qid, sid)]
                true_src_filler = src_sample['option1'] if src_sample['answer'] == '1' else src_sample['option2']
                false_src_filler = src_sample['option2'] if src_sample['answer'] == '1' else src_sample['option1']
                src_filler1 = true_src_filler if true_id == 1 else false_src_filler
                src_filler2 = false_src_filler if true_id == 1 else true_src_filler

                # Assign referents
                true_trans_ref1 = true_ref_tokens_true if true_id == 1 else false_ref_tokens_true
                true_trans_ref2 = false_ref_tokens_true if true_id == 1 else true_ref_tokens_true
                false_trans_ref1 = true_ref_tokens_false if true_id == 2 else false_ref_tokens_false
                false_trans_ref2 = false_ref_tokens_false if true_id == 2 else true_ref_tokens_false

                new_sample_contra = {'qID': '{:s}-{:d}'.format(qid, sid),
                                     'sentence': true_ref_annotations['ambiguous_sentence'],
                                     'translation1': translation1,
                                     'translation2': translation2,
                                     'answer': true_id,

                                     # Added for analysis
                                     'pronoun1': pronoun1,
                                     'pronoun2': pronoun2,
                                     'referent1_en': src_filler1,
                                     'referent2_en': src_filler2,
                                     'true_translation_referent_of_pronoun1_{:s}'.format(tgt_lang): true_trans_ref1,
                                     'true_translation_referent_of_pronoun2_{:s}'.format(tgt_lang): true_trans_ref2,
                                     'false_translation_referent_of_pronoun1_{:s}'.format(tgt_lang): false_trans_ref1,
                                     'false_translation_referent_of_pronoun2_{:s}'.format(tgt_lang): false_trans_ref2}
                contra_data.append(new_sample_contra)

                # Only possible for German
                if tgt_lang == 'de':
                    new_sample_mlm = {'qID': '{:s}-{:d}'.format(qid, sid),
                                      'sentence': true_ref_annotations['ambiguous_sentence'],
                                      'tgt_context': true_ref_tgt_context,
                                      'option1': pronoun1,
                                      'option2': pronoun2,
                                      'answer': true_id,

                                      # Added for analysis
                                      'referent1': src_filler1,
                                      'referent2': src_filler2,
                                      'context_referent_of_option1_{:s}'.format(tgt_lang): true_trans_ref1,
                                      'context_referent_of_option2_{:s}'.format(tgt_lang): true_trans_ref2}
                    mlm_data.append(new_sample_mlm)

            else:

                print('=' * 20)
                print('PRONOUN MATCH!')
                samples_with_matching_pronouns.add(qid)

        # Report
        if qid_id > 0 and (qid_id + 1) % 100 == 0:
            _report_stats()

    _report_stats(is_final=True)
    print('=' * 20)
    print('# samples for which alignments could not be found: {:d}'.format(len(samples_with_no_alignment)))
    print('# samples with incompatible morphological features: {:d}'.format(len(samples_with_bad_morphology)))
    print('# samples with bad translations: {:d}'.format(len(samples_with_bad_translations)))
    print('# samples with matching pronouns: {:d}'.format(len(samples_with_matching_pronouns)))

    # Save data to disc
    print('=' * 20)
    skipped = 0
    contra_out_path = out_path[:-6] + '_{:s}.en-{:s}.jsonl'.format('contra', tgt_lang)
    with open(contra_out_path, 'w', encoding='utf8') as cof:
        for sample_id, sample in enumerate(contra_data):
            if _skip_sample(sample['qID']):
                skipped += 1
                continue
            cof.write(json.dumps(sample, ensure_ascii=False))
            if sample_id < len(contra_data) - 1:
                cof.write('\n')
    print('Saved {:d} CONTRASTIVE samples to {:s}\n'.format(len(contra_data) - skipped, contra_out_path))

    print('=' * 20)
    skipped = 0
    mlm_out_path = out_path[:-6] + '_{:s}.en-{:s}.jsonl'.format('mlm', tgt_lang)
    with open(mlm_out_path, 'w', encoding='utf8') as mof:
        for sample_id, sample in enumerate(mlm_data):
            if _skip_sample(sample['qID']):
                skipped += 1
                continue
            mof.write(json.dumps(sample, ensure_ascii=False))
            if sample_id < len(mlm_data) - 1:
                mof.write('\n')
    print('Saved {:d} MLM samples to {:s}\n'.format(len(mlm_data) - skipped, mlm_out_path))

    print('=' * 20)
    skipped = 0
    mlm_nouns_out_path = out_path[:-6] + '_{:s}.en-{:s}.jsonl'.format('mlm_nouns', tgt_lang)
    with open(mlm_nouns_out_path, 'w', encoding='utf8') as mnof:
        for sample_id, sample in enumerate(mlm_nouns_data):
            if _skip_sample(sample['qID'], exclude_pronoun_match=True):
                skipped += 1
                continue
            mnof.write(json.dumps(sample, ensure_ascii=False))
            if sample_id < len(mlm_nouns_data) - 1:
                mnof.write('\n')
    print('Saved {:d} MLM-nouns samples to {:s}\n'.format(len(mlm_nouns_data) - skipped, mlm_nouns_out_path))

    print('=' * 20)
    skipped = 0
    qa_out_path = out_path[:-6] + '_{:s}.en-{:s}.jsonl'.format('qa', tgt_lang)
    with open(qa_out_path, 'w', encoding='utf8') as qof:
        for sample_id, sample in enumerate(qa_data):
            if _skip_sample(sample['qID'], exclude_pronoun_match=True):
                skipped += 1
                continue
            qof.write(json.dumps(sample, ensure_ascii=False))
            if sample_id < len(qa_data) - 1:
                qof.write('\n')
    print('Saved {:d} QA samples to {:s}\n'.format(len(qa_data) - skipped, qa_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_json_data_path', type=str, required=True,
                        help='path to the JSON file containing filtered, annotated WinoGrande samples')
    parser.add_argument('--source_sentences_path', type=str, required=True,
                        help='path to the text file containing the tokenized source language WinoGrande data')
    parser.add_argument('--target_sentences_path', type=str, required=True,
                        help='path to the text file containing the tokenized translated WinoGrande data')
    parser.add_argument('--qid_path', type=str, required=True,
                        help='path to the file containing IDs of translated source sentences')
    parser.add_argument('--alignments_path', type=str, required=True,
                        help='path to the file containing automatically learned alignments')
    parser.add_argument('--tgt_lang', type=str, required=True, choices=['de', 'fr', 'ru'],
                        help='Code corresponding to the target language of the translations')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the destination of the constructed dataset')
    parser.add_argument('--exclude_multi_sentence_samples', action='store_true',
                        help='whether to exclude multi-sentence samples from the derived datasets; recommended')
    args = parser.parse_args()

    # Initialize Stanza parser for the target language
    if args.tgt_lang != 'ru':
        nlp = stanza.Pipeline(args.tgt_lang, processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)
    else:
        nlp = stanza.Pipeline(args.tgt_lang, processors='tokenize,pos,lemma,depparse', use_gpu=True)

    # Initialize Stanza parser for the source language
    nlp_src = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)

    # Initialize Moses detokenizer
    detokenizer = MosesDetokenizer(lang='en')

    build_corpora(args.source_json_data_path,
                  args.source_sentences_path,
                  args.target_sentences_path,
                  args.qid_path,
                  args.alignments_path,
                  args.tgt_lang,
                  args.out_path,
                  args.exclude_multi_sentence_samples)

