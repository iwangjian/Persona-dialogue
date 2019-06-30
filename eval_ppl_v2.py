#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
"""Script for the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2019-ECDT) Task2.
This script evaluates the perplexity of the submitted model.
This uses a the version of the dataset which does not contain the "Golden Response" .
Leaderboard scores will be run in the same form but on a hidden test set.
The official vocabulary for the competition is based on using "jieba"
and is built on the training and validation sets. The test set contains some
tokens which are not in this dictionary--this tokens will not be provided, but
we will also *SKIP* calculating perplexity on these tokens. The model should
still produce a good guess for the remaining tokens in the sentence, so
handling unknown words or expanding the vocabulary with pre-trained or
multitasked embeddings are legitimate strategies that may or may not impact the
score of the models.

The model will be asked to predict one word at a time.
This requires each team to implement the following function:
def next_word_probability(self, context, partial_out):
    Return probability distribution over next words given a context and a partial true output.
    This is used to calculate the per-word perplexity.
    Arguments:
    context -- dialogue histories and personal profiles of every speaker
    partial_out -- list of previous "true" words
    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.
"""
from main import Model
import math
import json
import sys
import codecs


def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return [json.loads(i) for i in contents]


def eval_ppl(model, context, resp_gt, vocab):
    """
    Compute the perplexity for the model on the "Golden Responses"
    :param model: class, model class that have a method named 'next_word_probability'
    :param context: dict, given context
    :param resp_gt: list, list of tokens of the "Golden Responses"
    :param vocab: list, target vocabulary that the perplexity is evaluated on
    :return: list, [average loss, average perplexity]
             if a token in resp_gt is contained in vocab and receives 0 probability in the returned
             value of the method 'next_word_probability', then 'inf' will be returned
    """
    loss = 0
    num_tokens = 0
    num_unk = 0
    for i in range(len(resp_gt)):
        if resp_gt[i] in vocab:
            probs, eos_probs = model.next_word_probability(context, resp_gt[:i])
            prob_true = probs.get(resp_gt[i], 0)
            if prob_true > 0:
                prob_true /= (sum((probs.get(k, 0) for k in vocab)) + eos_probs)
                loss -= math.log(prob_true)
            else:
                loss = float('inf')
            num_tokens += 1
        else:
            num_unk += 1
    probs, eos_probs = model.next_word_probability(context, resp_gt)
    eos_probs /= (sum((probs.get(k, 0) for k in vocab)) + eos_probs)
    loss -= math.log(eos_probs)
    num_tokens += 1
    return loss / num_tokens, math.exp(loss / num_tokens)


if __name__ == '__main__':
    model = Model()
    if len(sys.argv) < 4:
        print('Too few args for this script')

    vocab_file = sys.argv[1]
    random_test = sys.argv[2]
    biased_test = sys.argv[3]

    with codecs.open(vocab_file, 'r', 'utf-8') as f:
        vocab = set([i.strip() for i in f.readlines() if len(i.strip()) != 0])
    random_test_data = read_dialog(random_test)
    biased_test_data = read_dialog(biased_test)

    random_ppl = 0
    biased_ppl = 0

    for count, dialog in enumerate(random_test_data):
        if count % 100 == 0:
            print(count)
        resp_gt = dialog['golden_response'][0].split()
        del dialog['golden_response']
        random_ppl += eval_ppl(model, dialog, resp_gt, vocab)[1]

    for count, dialog in enumerate(biased_test_data):
        if count % 100 == 0:
            print(count)
        resp_gt = dialog['golden_response'][0].split()
        del dialog['golden_response']
        biased_ppl += eval_ppl(model, dialog, resp_gt, vocab)[1]

    random_ppl /= len(random_test_data)
    biased_ppl /= len(biased_test_data)

    print('random ppl', random_ppl)
    print('biased ppl', biased_ppl)
    if random_ppl + biased_ppl == float('inf'):
        print('You model got an inf for PPL score, mostly likely you do not assign ' +
              'any probability to a token in the golden response. You should consider to enlarge your vocab')
    else:
        print((random_ppl + biased_ppl) / 2.0)
