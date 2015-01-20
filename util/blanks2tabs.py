#!/usr/bin/env python

'''
script to convert blanks to tabs
'''

__author__ = 'Erwin Marsi <e.c.marsi@uvt.nl>'
__version__ = '$Id: blanks2tabs.py,v 1.1 2006/01/05 17:42:36 erwin Exp $'

import sys
import string
import optparse
import codecs


def blanks2tabs(instream, outstream, encoding, replacement):
    stdinReader = codecs.lookup(options.encoding)[-2]
    stdoutWriter = codecs.lookup(options.encoding)[-1]
    
    instream = stdinReader(instream)
    outstream = stdoutWriter(outstream)
    
    replacement = replacement.encode(encoding)
    
    for l in instream:
        l = l.replace('\t',replacement)
        outstream.write(string.join(l.split(), '\t') + '\n')
            
# main stuff    

usage = \
"""
    %prog [options] <INFILE >OUTFILE

purpose:
    Converts each sequence of blanks to a single tab.
    Reads from standard input and writes to standard output.
    Expects input in tabular format with columns separated by one or more blanks.
    Tabs in column content are replaced by blanks (by default)."""

parser = optparse.OptionParser(usage, version=__version__)
                              
parser.add_option('-e', '--encoding',
                  dest='encoding', 
                  metavar='STRING', 
                  default='utf-8',
                  help="input and output character encoding (default is utf-8)")

parser.add_option('-t', '--tab-replace',
                  dest='replace',
                  default=' ',
                  metavar='STRING', 
                  help='replacement for tabs in column content (default is a single blank)')

(options, args) = parser.parse_args()

if '\t' in options.replace:
    sys.exit('Error: tab replacement string contains tab character!')

blanks2tabs(sys.stdin, 
            sys.stdout,  
            options.encoding,
            options.replace)


        