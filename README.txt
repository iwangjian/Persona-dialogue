Evaluation scripts should be put in a same folder with a file named "main.py"
This file should contain a class named "Model".
Each team should implement two methods for this class:


def next_word_probability(self, context, partial_out):
    Return probability distribution over next words given a context and a partial true output.
    This is used to calculate the per-word perplexity.
    Arguments:
    context -- dialogue histories and personal profiles of every speaker
    partial_out -- list of previous "true" words
    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.


def gen_response(self, contexts):
    return a list of responses for each context
    Arguments:
    contexts -- a list of context, each context contains dialogue histories and personal profiles of every speaker
    Returns a list, where each element is the response of the corresponding context

see the example file in this folder

Scripts usage:
-----------------------------------------------
eval_ppl.py: evaluate the perplexity of the submitted model

$ python3 eval_ppl.py vocab.txt test_data_biased.json test_data_random.json

vocab.txt: a word vocabulary, only the words contained in this vocabulary are counted
test_data_biased.json: biased test dataset. The dialogues in this dataset is filtered by human
test_data_random.json: random test dataset. The dialogues in this dataset is filtered using scripts
-----------------------------------------------
eval_bleu.py: evaluate the BLEU score of the submitted paper

$ python3 eval_bleu.py test_data_biased.json test_data_random.json

test_data_biased.json: biased test dataset. The dialogues in this dataset is filtered by human
test_data_random.json: random test dataset. The dialogues in this dataset is filtered using scripts
-----------------------------------------------
eval_distinct.py: evaluate the distinct score of the submitted paper

$ python3 eval_distinct.py test_data_biased.json test_data_random.json

test_data_biased.json: biased test dataset. The dialogues in this dataset is filtered by human
test_data_random.json: random test dataset. The dialogues in this dataset is filtered using scripts
