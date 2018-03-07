#! /bin/python3

import re
from collections import defaultdict

FILE = "/home/kilian/mnt/tulip/tmp/verbose_f1.log"
MAXLENGTH = 200
SMOOTH_RANGE = 5

pattern = re.compile(r"^\s*\d+\s+(\d+)\s+\d+\.\d+\s+\d+\.\d+\s+(\d+)\s+(\d+)\s+(\d+).*$")


def main():
    file = open(FILE, "r")
    stats = defaultdict(lambda: (0, 0, 0, 0))
    leq_stats = defaultdict(lambda: (0, 0, 0, 0))
    smooth_stats = defaultdict(lambda: (0, 0, 0, 0))
    lengths = {0}
    n_lines = 0
    for line in file:
        match = pattern.match(line)
        if match:
            # print(match.groups())
            length = int(match.group(1))
            matched_brackets = int(match.group(2))
            gold_brackets = int(match.group(3))
            cand_brackets = int(match.group(4))
            stats[length] = matched_brackets + stats[length][0], \
                            gold_brackets + stats[length][1], \
                            cand_brackets + stats[length][2], \
                            1 + stats[length][3]
            for le_length in range(length, MAXLENGTH):
                leq_stats[le_length] = matched_brackets + leq_stats[le_length][0], \
                                       gold_brackets + leq_stats[le_length][1], \
                                       cand_brackets + leq_stats[le_length][2], \
                                       1 + leq_stats[le_length][3]
            for smooth_length in range(length - SMOOTH_RANGE, length + SMOOTH_RANGE + 1):
                smooth_stats[smooth_length] = matched_brackets + smooth_stats[smooth_length][0], \
                                       gold_brackets + smooth_stats[smooth_length][1], \
                                       cand_brackets + smooth_stats[smooth_length][2], \
                                       1 + smooth_stats[smooth_length][3]

            lengths.add(length)
            n_lines += 1
        # if n_lines > 500:
        #     break

    def precicion(stat):
        return 0 if stat[1] == 0 else stat[0] / stat[2] * 100.0

    def recall(stat):
        return 0 if stat[2] == 0 else stat[0] / stat[1] * 100.0

    def f1(stat):
        prec = precicion(stat)
        rec = recall(stat)
        return 0 if prec + rec == 0 else prec * rec * 2 / (prec + rec)

    print("length", "count", "prec", "rec", "F1", "leq_count", "leq_F1", "smooth_count", "smooth_F1")
    for length in sorted(lengths):
        print(length,
              stats[length][3], precicion(stats[length]), recall(stats[length]), f1(stats[length]),
              leq_stats[length][3], f1(leq_stats[length]),
              smooth_stats[length][3], f1(smooth_stats[length])
              )


if __name__ == "__main__":
    main()