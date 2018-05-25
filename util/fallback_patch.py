#! /bin/python3
# based on https://stackoverflow.com/a/5266961

import sys
import os
import re

if "--help" in sys.argv or len(sys.argv) != 3:
    print("""Patches parse failures in first file with parses from second file. 
Expects two arguments: /path/to/primary/file /path/to/secondary/file.
File format: NEGRA export with 5 columns.
Result is written to stdout.""")
    exit()


match_start = re.compile(r'#BOS\s+(\d+)')
match_line = re.compile(r'([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+(\d+)')
match_trivial = re.compile(r'#500\s+S\s+--\s+--\s+0')
match_end = re.compile(r'#EOS\s+(\d+)')

with open(sys.argv[1]) as primary, open(sys.argv[2]) as secondary:
    sent = None
    line_buffer = []
    trivial = False
    function_tag = False
    while True:
        try:
            line = next(primary)
            if match_start.match(line):
                sent = int(match_start.match(line).group(1))
                line_buffer.append(line)
                continue
            if match_end.match(line):
                assert sent is not None and sent == int(match_end.match(line).group(1))
                line_buffer.append(line)
                if function_tag or not trivial:
                    for __line in line_buffer:
                        print(__line, end='')
                else:
                    sec_sent = None
                    while True:
                        sec_line = next(secondary)
                        if match_start.match(sec_line):
                            sec_sent = int(match_start.match(sec_line).group(1))
                            if sec_sent == sent:
                                print(sec_line, end='')
                        if sec_sent != sent:
                            continue
                        if match_end.match(sec_line):
                            print(sec_line, end='')
                            break
                        if match_line.match(sec_line):
                            print(sec_line, end='')
                sent = None
                line_buffer = []
                sent = None
                trivial = False
                function_tag = False
                continue
            if match_trivial.match(line):
                line_buffer.append(line)
                trivial = True
                continue
            if match_line.match(line):
                line_buffer.append(line)
                if match_line.match(line).group(4) != '--':
                    function_tag = True
                continue
        except StopIteration:
            break
