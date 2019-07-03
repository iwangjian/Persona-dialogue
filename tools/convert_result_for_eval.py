#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: convert_result_for_eval.py
"""

import sys
import json


def convert_result_for_eval(sample_file, result_file, output_file):
    """
    convert_result_for_eval
    """
    sample_list = [line.strip() for line in open(sample_file, 'r')]
    result_list = [line.strip() for line in open(result_file, 'r')]

    assert len(sample_list) == len(result_list)
    fout = open(output_file, 'w')
    for i, sample in enumerate(sample_list):
        sample = json.loads(sample, encoding="utf-8")
        response = sample["golden_response"][0]
        fout.write(result_list[i] + "\t\t" + response + "\n")

    fout.close()


def main():
    convert_result_for_eval(sys.argv[1],
                            sys.argv[2],
                            sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
