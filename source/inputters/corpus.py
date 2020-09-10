#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/inputters/corpus.py
"""

import os
import torch
import json
from tqdm import tqdm
from source.inputters.field import tokenize
from source.inputters.field import TextField
from source.inputters.field import NumberField
from source.inputters.dataset import Dataset


class Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 vocab_file=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        if vocab_file is not None:
            prepared_vocab_file = vocab_file
        else:
            prepared_vocab_file = data_prefix + "." + str(max_vocab_size) + ".vocab.pt"

        prepared_data_file = data_prefix + ".data_v2.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = {}
        self.padding_idx = None

    def load(self):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()

        self.load_data(self.prepared_data_file)

    def reload(self, data_type='test', data_file=None):
        """
        reload
        """
        if data_file is not None:
            data_raw = self.read_data(data_file, data_type="test")
            data_examples = self.build_examples(data_raw)
            self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self):
        """
        load_vocab
        """

        print("Loading prepared vocab from {} ...".format(self.prepared_vocab_file))
        vocab_dict = torch.load(self.prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size) 
                for name, field in self.fields.items() 
                    if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def build_examples(self, data, data_type=None):
        """
        Args
        ----
        data: ``List[Dict]``
        data_type: str
        """
        examples = []
        if data_type == 'test':
            for raw_data in data:
                example = {}
                for name, strings in raw_data.items():
                    example[name] = self.fields[name].numericalize(strings)
                examples.append(example)
        else:
            for raw_data in tqdm(data):
                example = {}
                for name, strings in raw_data.items():
                    example[name] = self.fields[name].numericalize(strings)
                examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        train_file = os.path.join(self.data_dir, "train.json")
        valid_file = os.path.join(self.data_dir, "valid.json")
        test_file = os.path.join(self.data_dir, "test.json")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class KnowledgeCorpus(Corpus):
    """
    KnowledgeCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 vocab_file=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False,
                 with_label=False):
        super(KnowledgeCorpus, self).__init__(data_dir=data_dir,
                                              data_prefix=data_prefix,
                                              min_freq=min_freq,
                                              max_vocab_size=max_vocab_size,
                                              vocab_file=vocab_file)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab
        self.with_label = with_label

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
            self.CUE = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)
            self.CUE = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        if self.with_label:
            self.INDEX = NumberField()
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'index': self.INDEX}
        else:
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}

        # load vocab
        if not os.path.exists(self.prepared_vocab_file):
            self.build()
        else:
            self.load_vocab()
        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

        def src_filter_pred(src):
            """
            src_filter_pred
            """
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            """
            tgt_filter_pred
            """
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])

    def read_data(self, data_file, data_type="train"):
        """
        read_data:q
        """
        num = 0
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                dialog = json.loads(line, encoding='utf-8')
                history = dialog["dialog"]
                uid = [int(i) for i in dialog["uid"]]
                profile = dialog["profile"]
                if "responder_profile" in dialog.keys():
                    responder_profile = dialog["responder_profile"]
                elif "response_profile" in dialog.keys():
                    responder_profile = dialog["response_profile"]
                else:
                    raise ValueError("No responder_profile or response_profile!")

                src = ""
                for idx, sent in zip(uid, history):
                    #tag_list = profile[idx]["tag"][0].split(';')
                    #loc_content = profile[idx]["loc"]
                    #tag_list.append(loc_content)
                    #tag_content = ' '.join(tag_list)
                    sent_content = sent[0]
                    #src += tag_content
                    #src += ' '
                    src += sent_content
                    src += ' '

                src = src.strip()
                tgt = dialog["golden_response"][0]
                filter_knowledge = []
                if type(responder_profile["tag"]) is list:
                    filter_knowledge.append(' '.join(responder_profile["tag"][0].split(';')))
                else:
                    filter_knowledge.append(' '.join(responder_profile["tag"].split(';')))
                filter_knowledge.append(responder_profile["loc"])
                data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge})

                num += 1
                if num < 10:
                    print("src:", src)
                    print("tgt:", tgt)
                    print("cue:", filter_knowledge)
                    print("\n")

        filtered_num = len(data)
        if not data_type == "test" and self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data
