#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
import codecs
import torch
import math
from source.inputters.corpus import KnowledgeCorpus
from source.inputters.dataset import Dataset
from source.models.seq2seq import Seq2Seq
from source.utils.generator import TopKGenerator


class Config:
    def __init__(self):
        """
        Init all the model configs
        """
        # Data
        self.data_dir = "data/"
        self.data_prefix = "ecdt2019"
        self.vocab_file = "vocab.pt"
        self.embed_file = None

        # Network
        self.embed_size = 200
        self.hidden_size = 512
        self.max_vocab_size = 30000
        self.min_len = 1
        self.max_len = 500
        self.num_layers = 2
        self.attn = 'general'   # ['none', 'mlp', 'dot', 'general']
        self.share_vocab = True
        self.bidirectional = True
        self.with_bridge = True
        self.tie_embedding = True
        self.use_gpu = torch.cuda.is_available()
        self.gpu = 0
        self.dropout = 0.3

        # Testing
        self.ckpt = "models/best.model"
        self.beam_size = 4
        self.max_dec_len = 40
        self.ignore_unk = True
        self.length_average = True
        

class Model:
    """
    This is an example model. It reads predefined dictionary and predict a fixed distribution.
    For a correct evaluation, each team should implement 3 functions:
    next_word_probability
    gen_response
    """
    def __init__(self):
        """
        Init whatever you need here
        """
        vocab_file = 'data/vocab.txt'
        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            vocab = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
        self.vocab = vocab
        self.freqs = dict(zip(self.vocab[::-1], range(len(self.vocab))))

        # Our code are as follows
        config = Config()
        torch.cuda.set_device(device=config.gpu)
        self.config = config

        # Data definition
        self.corpus = KnowledgeCorpus(data_dir=config.data_dir, data_prefix=config.data_prefix,
                                      min_freq=0, max_vocab_size=config.max_vocab_size,
                                      vocab_file=config.vocab_file,
                                      min_len=config.min_len, max_len=config.max_len,
                                      embed_file=config.embed_file, share_vocab=config.share_vocab)
        # Model definition
        self.model = Seq2Seq(src_vocab_size=self.corpus.SRC.vocab_size,
                             tgt_vocab_size=self.corpus.TGT.vocab_size,
                             embed_size=config.embed_size, hidden_size=config.hidden_size,
                             padding_idx=self.corpus.padding_idx,
                             num_layers=config.num_layers, bidirectional=config.bidirectional,
                             attn_mode=config.attn, with_bridge=config.with_bridge,
                             tie_embedding=config.tie_embedding, dropout=config.dropout,
                             use_gpu=config.use_gpu)
        print(self.model)
        self.model.load(config.ckpt)

        # Generator definition
        self.generator = TopKGenerator(model=self.model, src_field=self.corpus.SRC,
                                       tgt_field=self.corpus.TGT, cue_field=self.corpus.CUE,
                                       beam_size=config.beam_size, max_length=config.max_dec_len,
                                       ignore_unk=config.ignore_unk,
                                       length_average=config.length_average, use_gpu=config.use_gpu)
        self.BOS = self.generator.BOS
        self.EOS = self.generator.EOS
        self.stoi = self.corpus.SRC.stoi
        self.itos = self.corpus.SRC.itos

    def next_word_probability(self, context, partial_out):
        """
        Return probability distribution over next words given a partial true output.
        This is used to calculate the per-word perplexity.

        :param context: dict, contexts containing the dialogue history and personal
                        profile of each speaker
                        this dict contains following keys:

                        context['dialog']: a list of string, dialogue histories (tokens in each utterances
                                           are separated using spaces).
                        context['uid']: a list of int, indices to the profile of each speaker
                        context['profile']: a list of dict, personal profiles for each speaker
                        context['responder_profile']: dict, the personal profile of the responder

        :param partial_out: list, previous "true" words
        :return: a list, the first element is a dict, where each key is a word and each value is a probability
                         score for that word. Unset keys assume a probability of zero.
                         the second element is the probability for the EOS token

        e.g.
        context:
        { "dialog": [ ["How are you ?"], ["I am fine , thank you . And you ?"] ],
          "uid": [0, 1],
          "profile":[ { "loc":"Beijing", "gender":"male", "tag":"" },
                      { "loc":"Shanghai", "gender":"female", "tag":"" } ],
          "responder_profile":{ "loc":"Beijing", "gender":"male", "tag":"" }
        }

        partial_out:
        ['I', 'am']

        ==>  {'fine': 0.9}, 0.1
        """
        test_raw = self.read_data(context)
        test_data = self.corpus.build_examples(test_raw, data_type='test')
        dataset = Dataset(test_data)
        data_iter = dataset.create_batches(batch_size=1, shuffle=False, device=self.config.gpu)
        inputs = None
        for batch in data_iter:
            inputs = batch
            break

        partial_out_idx = [self.stoi[s] if s in self.stoi.keys() else self.stoi['<unk>'] for s in partial_out]

        # switch the model to evaluate mode
        self.model.eval()
        with torch.no_grad():
            enc_outputs, dec_init_state = self.model.encode(inputs)
            long_tensor_type = torch.cuda.LongTensor if self.config.use_gpu else torch.LongTensor

            # Initialize the input vector
            input_var = long_tensor_type([self.BOS] * 1)
            # Inflate the initial hidden states to be of size: (1, H)
            dec_state = dec_init_state.inflate(1)

            for t in range(len(partial_out_idx)):
                # Run the RNN one step forward
                output, dec_state, attn = self.model.decode(input_var, dec_state)
                input_var = long_tensor_type([partial_out_idx[t]])

            output, dec_state, attn = self.model.decode(input_var, dec_state)
            log_softmax_output = output.squeeze(1)
        log_softmax_output = log_softmax_output.cpu().numpy()
        prob_output = [math.exp(i) for i in log_softmax_output[0]]

        # The first 4 tokens are: '<pad>' '<unk>' '<bos>' '<eos>'
        freq_dict = {}
        for i in range(4, len(self.itos)):
            freq_dict[self.itos[i]] = prob_output[i]
        eos_prob = prob_output[3]
        return freq_dict, eos_prob

    def gen_response(self, contexts):
        """
        Return a list of responses to each context.

        :param contexts: list, a list of context, each context is a dict that contains the dialogue history and personal
                         profile of each speaker
                         this dict contains following keys:

                         context['dialog']: a list of string, dialogue histories (tokens in each utterances
                                            are separated using spaces).
                         context['uid']: a list of int, indices to the profile of each speaker
                         context['profile']: a list of dict, personal profiles for each speaker
                         context['responder_profile']: dict, the personal profile of the responder

        :return: list, responses for each context, each response is a list of tokens.

        e.g.
        contexts:
        [{ "dialog": [ ["How are you ?"], ["I am fine , thank you . And you ?"] ],
          "uid": [0, 1],
          "profile":[ { "loc":"Beijing", "gender":"male", "tag":"" },
                      { "loc":"Shanghai", "gender":"female", "tag":"" } ],
          "responder_profile":{ "loc":"Beijing", "gender":"male", "tag":"" }
        }]

        ==>  [['I', 'am', 'fine', 'too', '!']]
        """
        test_raw = self.read_data(contexts[0])
        test_data = self.corpus.build_examples(test_raw, data_type='test')
        dataset = Dataset(test_data)
        data_iter = dataset.create_batches(batch_size=1, shuffle=False, device=self.config.gpu)
        results = self.generator.generate(batch_iter=data_iter)
        res = [result.preds[0].split(" ") for result in results]
        return res

    @staticmethod
    def read_data(dialog):
        history = dialog["dialog"]
        uid = [int(i) for i in dialog["uid"]]
        if "responder_profile" in dialog.keys():
            responder_profile = dialog["responder_profile"]
        elif "response_profile" in dialog.keys():
            responder_profile = dialog["response_profile"]
        else:
            raise ValueError("No responder_profile or response_profile!")

        src = ""
        for idx, sent in zip(uid, history):
            sent_content = sent[0]
            src += sent_content
            src += ' '

        src = src.strip()
        tgt = "NULL"
        filter_knowledge = []
        if type(responder_profile["tag"]) is list:
            filter_knowledge.append(' '.join(responder_profile["tag"][0].split(';')))
        else:
            filter_knowledge.append(' '.join(responder_profile["tag"].split(';')))
        filter_knowledge.append(responder_profile["loc"])
        data = [{'src': src, 'tgt': tgt, 'cue': filter_knowledge}]
        return data
