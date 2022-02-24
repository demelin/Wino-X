import argparse

import mxnet as mx
import numpy as np

from mlm.scorers import MLMScorer
from mlm.models import get_pretrained


ANIMATE_TESTS_SG = ['The _ is eating a meal.',
                    'The _ is working all week.',
                    'The _ is sleeping at night.',
                    'The _ is crying out of sadness.',
                    'The _ is laughing out of joy.',
                    'The _ is talking to friends.',
                    'The _ is feeling ill.',
                    'The _ has happy thoughts.',
                    'The _ relaxes after a long day.',
                    'The _ loves ice cream.',
                    'The _ does not want to die.',
                    'The _ knows how to whistle.',
                    'The _ has a big family.',
                    'The _ is a person.',
                    'The _ is alive.']

ANIMATE_TESTS_PL = ['The _ are eating a meal.',
                    'The _ are working all week.',
                    'The _ are sleeping at night.',
                    'The _ are crying out of sadness.',
                    'The _ are laughing out of joy.',
                    'The _ are talking to friends.',
                    'The _ are feeling ill.',
                    'The _ have happy thoughts.',
                    'The _ relax after a long day.',
                    'The _ love ice cream.',
                    'The _ do not want to die.',
                    'The _ know how to whistle.',
                    'The _ have a big family.',
                    'The _ are people.',
                    'The _ are alive.']


INANIMATE_TESTS_SG = ['The _ is not very durable.',
                      'The _ is in somebody\'s possession.',
                      'The _ has never been cleaned.',
                      'The _ was made only recently.',
                      'The _ should be replaced with a newer one.',
                      'The _ has a large area.',
                      'The _ is very simple.',
                      'The _ is closed for the day.',
                      'The _ is open every week.',
                      'The _ has a lot of space.',
                      'The _ was founded long ago.',
                      'The _ is a thing.',
                      'The _ is a concept',
                      'The _ is an object.',
                      'The _ is a place.']

INANIMATE_TESTS_PL = ['The _ are not very durable.',
                      'The _ are in somebody\'s possession.',
                      'The _ have never been cleaned.',
                      'The _ were made only recently.',
                      'The _ should be replaced with a newer one.',
                      'The _ have a large area.',
                      'The _ are very simple.',
                      'The _ are closed for the day.',
                      'The _ are open every week.',
                      'The _ have a lot of space.',
                      'The _ were founded long ago.',
                      'The _ are things.',
                      'The _ are concepts',
                      'The _ are objects.',
                      'The _ are places.']


ANIMAL_TESTS_SG = ['The _ has many legs.',
                   'The _ loves to run wild.',
                   'The _ likes to hunt prey.',
                   'The _ cannot be tamed.',
                   'The _ likes to play fetch.',
                   'The _ is fast and strong.',
                   'The _ must be fed on time.',
                   'The _ makes a lot of noise.',
                   'The _ sleeps through most of the day.',
                   'The _ has fur and claws.',
                   'The _ is very territorial.',
                   'The _ lives in the jungle.',
                   'The _ lives in the forest.',
                   'The _ must be kept on a leash.',
                   'The _ is an animal.']

ANIMAL_TESTS_PL = ['The _ have many legs.',
                   'The _ love to run wild.',
                   'The _ like to hunt prey.',
                   'The _ cannot be tamed.',
                   'The _ like to play fetch.',
                   'The _ are fast and strong.',
                   'The _ must be fed on time.',
                   'The _ make a lot of noise.',
                   'The _ sleep through most of the day.',
                   'The _ have fur and claws.',
                   'The _ are very territorial.',
                   'The _ live in the jungle.',
                   'The _ live in the forest.',
                   'The _ must be kept on a leash.',
                   'The _ are animals.']


DEF_ANIMATE_SG = 'person'
DEF_ANIMATE_PL = 'people'

DEF_INANIMATE_SG = 'thing'
DEF_INANIMATE_PL = 'things'

DEF_ANIMAL_SG = 'animal'
DEF_ANIMAL_PL = 'animals'


def _fill_in_gaps(sentence_list, filler):
    """ Helper function for filling-in sentence gaps. """
    filled_sentence_list = list()
    for s in sentence_list:
        s_tokens = s.split()
        s_tokens[s_tokens.index('_')] = filler
        filled_sentence_list.append(' '.join(s_tokens))
    return filled_sentence_list


def test_if_animate(test_item,
                    is_singular,
                    lm_scorer,
                    default_scores_dict,
                    verbose=False):

    """ Estimates whether the input item (either a word or a phrase) is animate or not. """

    # Select set of sentences based on test item number
    ANIMATE_TESTS = ANIMATE_TESTS_SG if is_singular else ANIMATE_TESTS_PL
    INANIMATE_TESTS = INANIMATE_TESTS_SG if is_singular else INANIMATE_TESTS_PL
    ANIMAL_TESTS = ANIMAL_TESTS_SG if is_singular else ANIMAL_TESTS_PL

    DEF_ANIMATE = DEF_ANIMATE_SG if is_singular else DEF_ANIMATE_PL
    DEF_INANIMATE = DEF_INANIMATE_SG if is_singular else DEF_INANIMATE_PL
    DEF_ANIMAL= DEF_ANIMAL_SG if is_singular else DEF_ANIMAL_PL

    default_animate_key = 'default_animate_score_sg' if is_singular else 'default_animate_score_pl'
    default_inanimate_key = 'default_inanimate_score_sg' if is_singular else 'default_inanimate_score_pl'
    default_animal_key = 'default_animal_score_sg' if is_singular else 'default_animal_score_pl'

    if default_scores_dict[default_animate_key] is None:
        # Fill gaps with default animate item
        filled_default_animate_tests = _fill_in_gaps(ANIMATE_TESTS, DEF_ANIMATE)
        # Compute pseudo-NLL scores for the filled test sentences
        default_animate_scores = lm_scorer.score_sentences(filled_default_animate_tests)
        default_scores_dict[default_animate_key] = np.mean(default_animate_scores)

        if verbose:
            print('-' * 20)
            print(filled_default_animate_tests)
            print(default_animate_scores, default_scores_dict[default_animate_key])

    if default_scores_dict[default_inanimate_key] is None:
        # Fill gaps with default animate item
        filled_default_inanimate_tests = _fill_in_gaps(INANIMATE_TESTS, DEF_INANIMATE)
        # Compute pseudo-NLL scores for the filled test sentences
        default_inanimate_scores = lm_scorer.score_sentences(filled_default_inanimate_tests)
        default_scores_dict[default_inanimate_key] = np.mean(default_inanimate_scores)

        if verbose:
            print('-' * 20)
            print(filled_default_inanimate_tests)
            print(default_inanimate_scores, default_scores_dict[default_inanimate_key])

    if default_scores_dict[default_animal_key] is None:
        # Fill gaps with default animate item
        filled_default_animal_tests = _fill_in_gaps(ANIMAL_TESTS, DEF_ANIMAL)
        # Compute pseudo-NLL scores for the filled test sentences
        default_animal_scores = lm_scorer.score_sentences(filled_default_animal_tests)
        default_scores_dict[default_animal_key] = np.mean(default_animal_scores)

        if verbose:
            print('-' * 20)
            print(filled_default_animal_tests)
            print(default_animal_scores, default_scores_dict[default_animal_key])

    # Check whether the evaluated item is more likely to be animate or inanimate
    test_item = test_item.strip().lower()
    filled_animate_tests = _fill_in_gaps(ANIMATE_TESTS, test_item)
    test_animate_scores = lm_scorer.score_sentences(filled_animate_tests)
    test_animate_score = np.mean(test_animate_scores)

    if verbose:
        print('-' * 20)
        print(filled_animate_tests)
        print(test_animate_scores, test_animate_score)

    filled_inanimate_tests = _fill_in_gaps(INANIMATE_TESTS, test_item)
    test_inanimate_scores = lm_scorer.score_sentences(filled_inanimate_tests)
    test_inanimate_score = np.mean(test_inanimate_scores)

    if verbose:
        print('-' * 20)
        print(filled_inanimate_tests)
        print(test_inanimate_scores, test_inanimate_score)

    animate_diff = abs(default_scores_dict[default_animate_key]) - abs(test_animate_score)
    inanimate_diff = abs(default_scores_dict[default_inanimate_key]) - abs(test_inanimate_score)

    if verbose:
        print('-' * 20)
        print(animate_diff, inanimate_diff)

    if animate_diff > inanimate_diff:

        # Check if test item refers to an animal
        filled_animal_tests = _fill_in_gaps(ANIMAL_TESTS, test_item)
        test_animal_scores = lm_scorer.score_sentences(filled_animal_tests)
        test_animal_score = np.mean(test_animal_scores)
        animal_diff = abs(default_scores_dict[default_animal_key]) - abs(test_animal_score)

        if verbose:
            print('-' * 20)
            print(filled_animal_tests)
            print(test_animal_scores, test_animal_score, animal_diff)

        if animal_diff > animate_diff:
            return True, True, default_scores_dict
        else:
            return True, False, default_scores_dict
    else:
        return False, False, default_scores_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_item', type=str, required=True,
                        help='word / phrase to be evaluated')
    parser.add_argument('--is_singular', action='store_true',
                        help='denotes whether test item is singular')
    parser.add_argument('--verbose', action='store_true',
                        help='denotes whether output debugging info')
    parser.add_argument('--default_scores_dict', type=float, default=None,
                        help='dictionary holding default scores')
    args = parser.parse_args()

    ctxs = [mx.cpu()]
    model, vocab, tokenizer = get_pretrained(ctxs, 'roberta-large-en-cased')
    scorer = MLMScorer(model, vocab, tokenizer, ctxs)

    if args.default_scores_dict is None:
        args.default_scores_dict = {'default_animate_score_sg': None,
                                    'default_inanimate_score_sg': None,
                                    'default_animal_score_sg': None,
                                    'default_animate_score_pl': None,
                                    'default_inanimate_score_pl': None,
                                    'default_animal_score_pl': None}

    out = test_if_animate(args.test_item,
                          args.is_singular,
                          scorer,
                          args.default_scores_dict,
                          verbose=args.verbose)

    print('{:s} is animate: {} | is animal: {}'.format(args.test_item, out[0], out[1]))
