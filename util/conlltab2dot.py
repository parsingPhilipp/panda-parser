#!/usr/bin/env python

'''
script to convert data in CoNLL-X tabular format to dot graphs
'''

__author__ = 'Erwin Marsi <e.c.marsi@uvt.nl>'
__version__ = '$Id: conlltab2dot.py,v 1.2 2006/01/06 11:44:50 erwin Exp $'

import sys
import string
import optparse
import codecs


def conlltab2dot(instream, outstream, options):
    try:
        sentrange = expand(options.range)
    except:
        sys.exit('Error: illegal range specification: %s' % options.range)
        
    stdinReader = codecs.lookup(options.encoding)[-2]
    stdoutWriter = codecs.lookup(options.encoding)[-1]
    
    instream = stdinReader(instream)
    outstream = stdoutWriter(outstream)
    
    if options.shape == 'h':
        hierarchical(instream, outstream, sentrange)
    elif options.shape == 'l':
        linear(instream, outstream, sentrange)
    
    
def hierarchical(instream, outstream, sentrange):
    graph = ''
    label=''
    sno = 1
    
    for l in instream:
        l = l.strip()
        if l and sno in sentrange:
            id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel = l.split()
            form = form.replace('"','\\"')
            # for arabic
            form = form.replace('{','\\{')
            graph += 'n%s [label="%s"];\n' % (id, form)
            graph += 'n%s -> n%s [label="%s"];\n' % (id, head, deprel)
            label += '%s ' % form
        else:
            if graph:
                outstream.write('digraph s%d {\n' % sno + 
                                'rankdir=TB;\n'
                                'ordering=in;\n'
                                'node [shape=record];\n'
                                'n0 [label=ROOT];\n' + 
                                graph +
                                'label="(%d) %s";\n' % (sno, label) +
                                '}\n')
                graph = ''
                label = ''
            sno += 1

            
def linear(instream, outstream, sentrange):
    graph = ''
    label=''
    sno = 1
    
    for l in instream:
        l = l.strip()
        if l and sno in sentrange:
            id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel = l.split()
            form = form.replace('"','\\"')
            graph += 'n%s [label="%s"];\n' % (id, form)
            graph += 'n%s -> n%s [label="%s",weight=0];\n' % (id, head, deprel)
            graph += 'n%s -> n%s [style="dotted",arrowhead="none"];\n' % (int(id)-1, id)
            label += '%s ' % form
        else:
            if graph:
                outstream.write('digraph s%d {\n' % sno + 
                                'rankdir=LR;\n'
                                'ordering=in;\n'
                                'node [shape=record,width=0.01];\n'
                                'n0 [label=ROOT];\n' + 
                                graph +
                                'label="(%d) %s";\n' % (sno, label) +
                                '}\n')
                graph = ''
                label = ''
            sno += 1
            
def expand(spec, last=10000):
    '''
    expand a range specification to actual sentence numbers
    
    e.g. 1,3,5 => [1,2,5]
         2-6 => [2,3,4,5,6]
         -3 => [1,2,3]
         10- => [10,11,12,13,...,last]
    '''
    # last=10000 is not very elegant, 
    # but I don't know the size in advance :-( 
    sentno = {}
    
    for el in spec.split(','):
        fields = el.split('-')
        if len(fields) == 1:
            sentno[int(fields[0])] = None
        elif len(fields) == 2:
            start, end = 1, last
            
            if fields[0]:
                start = int(fields[0])
            else:
                start = 1
                
            if fields[1]:
                end = int(fields[1])
            else:
                end = last
            
            sentno.update(dict.fromkeys(range(start,end + 1)))
           
    sentno = sentno.keys()
    sentno.sort()
    return sentno



# main stuff    

usage = \
"""
    %prog [options] <INFILE >OUTFILE
     
purpose:
    Converts dependency structures in CoNLL-X tabular format 
    to dot graph specifications.
    Reads from standard input and writes to standard output.

examples:

   To generate postscript output on Linux, try:
 
      ./conlltab2dot.py <sample.conll | \\
      dot -Tps2 > /tmp/sample.ps && \\
      gv /tmp/sample.ps

    To generate PDF output on Mac OS X, try:
    
      ./conlltab2dot.py <sample.conll | \\
      /Applications/Graphviz.app/Contents/MacOS/dot -Tepdf | \\
      open -f -a preview
    """

parser = optparse.OptionParser(usage, version=__version__)
                              

parser.add_option('-e', '--encoding',
                  dest='encoding', 
                  metavar='STRING', 
                  default='utf-8',
                  help="character encoding (default is utf-8)")

parser.add_option('-r', '--range',
                  dest='range',
                  default='1-',
                  metavar='STRING', 
                  help='sentence range as comma-separated list of sentence numbers, '
                  'optionally using n-, -n, or n-m to denote inclusive ranges '
                  '(default is all sentences)')

parser.add_option('-s', '--shape',
                  dest='shape', 
                  choices=['h','l'],
                  default='h',
                  metavar='CHAR',
                  type='choice',
                  help="shape of graph where 'h' means hierarchical "
                  "and 'l' means linear (default is 'h')")

(options, args) = parser.parse_args()

conlltab2dot(sys.stdin, sys.stdout, options)


