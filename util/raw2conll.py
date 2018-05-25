#! /bin/python3
import sys

for line in sys.stdin:
    for idx, token in enumerate(line.split()):
        print(str(idx+1) + "\t" + token)
    print()
