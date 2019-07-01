#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: convert_session_to_sample.py
"""

import sys
import json
import random


def convert_session_to_sample(session_file, train_file, dev_file, dev_num=10000):
    """
    convert_session_to_sample
    """
    data_samples = []
    with open(session_file, 'r') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8")
            sample = dict()
            # here we only use single-turn dialog
            sample["dialog"] = session["dialog"][:-1]
            sample["uid"] = session["uid"][:-1]
            sample["profile"] = session["profile"]
            responder_uid = session["uid"][-1]
            sample["responder_profile"] = session["profile"][responder_uid]
            sample["golden_response"] = session["dialog"][-1]
            data_samples.append(sample)
            if i > 0 and i % 100000 == 0:
                print("read %d" % i)

    # randomly pick dev_num samples as dev data
    idxs = list(range(len(data_samples)))
    dev_idx = random.sample(idxs, dev_num)

    with open(train_file, 'w', encoding='utf-8') as ftrain, open(dev_file, 'w', encoding='utf-8') as fdev:
        for i, sample in enumerate(data_samples):
            line = json.dumps(sample, ensure_ascii=False)
            if i in dev_idx:
                fdev.write(line + '\n')
            else:
                ftrain.write(line + '\n')
            if i > 0 and i % 100000 == 0:
                print("writing %d" % i)


def main():
    convert_session_to_sample(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
