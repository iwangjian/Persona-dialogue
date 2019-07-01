#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
"""Script for the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2019-ECDT) Task2.
This script evaluates the corpus BLEU of the submitted model.
This uses a the version of the dataset which does not contain the "Golden Response" .
Leaderboard scores will be run in the same form but on a hidden test set.

This requires each team to implement the following function:
def gen_response(self, contexts):
    return a list of responses for each context
    Arguments:
    contexts -- a list of context, each context contains dialogue histories and personal profiles of every speaker
    Returns a list, where each element is the response of the corresponding context
"""
from __future__ import division
import math
import sys
import fractions
from collections import Counter
from itertools import chain
from main import Model
import json
import codecs

"""BLEU score implementation copied from NLTK 3.4.3"""
try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    class Fraction(fractions.Fraction):
        def __new__(cls, numerator=0, denominator=None, _normalize=True):
            cls = super(Fraction, cls).__new__(cls, numerator, denominator)
            # To emulate fraction.Fraction.from_float across Python >=2.7,
            # check that numerator is an integer and denominator is not None.
            if not _normalize and type(numerator) == int and denominator:
                cls._numerator = numerator
                cls._denominator = denominator
            return cls


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n-1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None, auto_reweigh=False):
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), (
        "The number of hypotheses and their reference(s) should be the " "same "
    )

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(
        p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    )
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))
    return s


def modified_precision(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def closest_ref_length(references, hyp_len):
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
    (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)


class SmoothingFunction:
    def __init__(self, epsilon=0.1, alpha=5, k=5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method1(self, p_n, *args, **kwargs):
        """
        Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
        """
        return [(p_i.numerator + self.epsilon) / p_i.denominator
                if p_i.numerator == 0 else p_i for p_i in p_n]


def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return [json.loads(i) for i in contents]


def eval_bleu(ref_resp, hyps_resp):
    """
    compute corpus BLEU score for the hyps_resp
    :param ref_resp: list, a list of reference list
    :param hyps_resp: list, a list of hyps list
    :return: average BLEU score for 1, 2, 3, 4-gram
    """
    if len(hyps_resp) == 0 or len(hyps_resp) == 0 or len(hyps_resp) != len(ref_resp):
        print("ERROR, eval_bleu get empty input or un-match inputs")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    return corpus_bleu(ref_resp, hyps_resp, smoothing_function=SmoothingFunction().method1)


if __name__ == '__main__':
    model = Model()

    if len(sys.argv) < 3:
        print('Too few args for this script')

    random_test = sys.argv[1]
    biased_test = sys.argv[2]
    random_test_data = read_dialog(random_test)
    biased_test_data = read_dialog(biased_test)
    random_ref_resp = [[j.split() for j in i['golden_response']] for i in random_test_data]
    biased_ref_resp = [[j.split() for j in i['golden_response']] for i in biased_test_data]

    for i in random_test_data:
        if 'golden_response' in i:
            del i['golden_response']

    for i in biased_test_data:
        if 'golden_response' in i:
            del i['golden_response']

    # random_hyps_resp = model.gen_response(random_test_data)
    # biased_hyps_resp = model.gen_response(biased_test_data)
    random_hyps_resp = []
    for count, i in enumerate(random_test_data):
        if count > 0 and count % 100 == 0:
            print(count)
        random_hyps_resp += model.gen_response([i])

    biased_hyps_resp = []
    for count, i in enumerate(biased_test_data):
        if count > 0 and count % 100 == 0:
            print(count)
        biased_hyps_resp += model.gen_response([i])

    random_bleu = eval_bleu(random_ref_resp, random_hyps_resp)
    biased_bleu = eval_bleu(biased_ref_resp, biased_hyps_resp)
    print('random BLEU', random_bleu)
    print('biased BLEU', biased_bleu)
    print((random_bleu + biased_bleu) / 2.0)
