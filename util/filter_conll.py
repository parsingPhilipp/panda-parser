#! /bin/python3

import sys
import re


def FILTER(x):
    return x % 10 > 0


fd = sys.stdin
regex = re.compile(r'(\d+)_\d+\s+.*')
empty = True

for line in fd:
    match = regex.match(line)
    if match:
        sent_id = int(match.group(1))
        if FILTER(sent_id):
            print(line, end='')
            empty = False
    else:
        if not empty:
            print(line, end='')
            empty = True