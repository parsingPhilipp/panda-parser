#! /bin/python3
# based on https://stackoverflow.com/a/5266961

import sys
from collections import defaultdict
import os

if "--help" in sys.argv or len(sys.argv) != 3:
    print("Compute TAGGING accuracy. Expects two arguments: /path/to/gold/standard /path/to/system/output.\n"
          "File format: one sentence per line. Word and pos are separated by '/'.")

testFile = open(sys.argv[1])
taggedFile = open(sys.argv[2])

finished = False
totals = defaultdict(lambda: defaultdict(lambda: 0))

correct = 0
in_total = 0

while not finished:
    trueLine = testFile.readline()
    if not trueLine: # end of file
        finished = True
    else:
        trueLine = trueLine.split() # tokenise by whitespace
        taggedLine = taggedFile.readline()
        if not taggedLine:
            raise Exception('Error: files are out of sync.')
        taggedLine = taggedLine.split()
        if len(trueLine) != len(taggedLine):
            raise Exception('Error: files are out of sync.')
        for i in range(len(trueLine)):
            truePair = trueLine[i].rsplit('/', 1)
            taggedPair = taggedLine[i].rsplit('/', 1)
            if truePair[0] != taggedPair[0]: # the words should match
                raise Exception('Error: files are out of sync.')
            trueTag = truePair[1]
            guessedTag = taggedPair[1]
            if trueTag == guessedTag:
                totals[trueTag]['truePositives'] += 1
                correct += 1
            else:
                totals[trueTag]['falseNegatives'] += 1
                totals[guessedTag]['falsePositives'] += 1
            in_total += 1

for tag in totals:
    print(tag, totals[tag]["truePositives"], totals[tag]["falseNegatives"], totals[tag]["falsePositives"])

print("In total", correct, "/", in_total, "=", correct / in_total)
