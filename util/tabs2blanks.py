#!/usr/bin/env python

'''
script to convert tabs to blanks
'''

__author__ = 'Erwin Marsi <e.c.marsi@uvt.nl>'
__version__ = '$Id: tabs2blanks.py,v 1.2 2006/01/05 17:42:36 erwin Exp $'

import sys
import string
import optparse
import codecs

def tabs2blanks(instream, outstream, maxwidth, encoding, replacement):
    stdinReader = codecs.lookup(options.encoding)[-2]
    stdoutWriter = codecs.lookup(options.encoding)[-1]
    
    instream = stdinReader(instream)
    outstream = stdoutWriter(outstream)
    
    colwidths = []
    
    for l in instream:
        for i, field in enumerate(l.split('\t')[:-1]):
            try:
                colwidths[i] = min( max(colwidths[i], len(field)), maxwidth )
            except IndexError:
                colwidths.append(len(field))
    
    format = string.join([u'%-' + str(w) + 's ' for w in colwidths]) + u' %s'
        
    instream.seek(0)
    
    replacement = replacement.encode(encoding)
    
    for l in instream:
        if l.strip():
            l = l.replace(' ', replacement) 
            try:
                outstream.write(format % tuple(l.split('\t')))
            except TypeError:
                sys.exit('Error: number of columns changes at line:\n' + l)
        else:
            outstream.write(l)
            

# main stuff    

usage = \
"""
    %prog [options] <INFILE >OUTFILE

purpose:
    Converts tabs to blanks in an attempt to align the column content.
    Reads from standard input and writes to standard output.
    Expects input in tabular format with columns separated by tabs.
    Spaces in column content are replaced by tabs (by default)."""

parser = optparse.OptionParser(usage, version=__version__)
                             
parser.add_option('-b', '--blank-replace',
                  dest='replace',
                  default='\t',
                  metavar='STRING', 
                  help='replacement for blanks in column content (default is tab)')

parser.add_option('-e', '--encoding',
                  dest='encoding', 
                  metavar='STRING', 
                  default='utf-8',
                  help="input and output character encoding (default is utf-8)")

parser.add_option('-m', '--max-width',
                  dest='maxwidth',
                  metavar='INT', 
                  default=sys.maxint,
                  type='int',
                  help='maximum width of a column (default is unlimited)')

(options, args) = parser.parse_args()

if ' ' in options.replace:
    sys.exit('Error: blank replacement string contains blank character!')
    
tabs2blanks(sys.stdin, 
            sys.stdout,  
            options.maxwidth,
            options.encoding,
            options.replace)


        