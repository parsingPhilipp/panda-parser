Readme to Version 2018-05-02
============================

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 2 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see \<<http://www.gnu.org/licenses/>\>.

### List of contributors

-   Kilian Gebhardt
-   Mark-Jan Nederhof
-   Markus Teichmann
-   Johann Seltmann
-   Sebastian Mielke
-   Kevin Mitlöhner


------------------------------------------------------------------------

## Overview

This project contains software that is developed in the context of research
on *hybrid grammars* and grammar-based approaches to 
natural language processing in general. 

Currently implemented grammar types:
- hybrid grammars coupling *linear context-free rewriting systems* (LCFRS) 
  and *(simple) definite clause programs* ((s)DCP)
- plain LCFRS
- aligned LCFRS/graph grammars (aligned hypergraph bimorphism)
- subclasses of LCFRS, such as *context-free grammars* (CFG) and *finite state automata* (FSA)

Functionality:
  - grammar induction from a corpus
  - parsing of sentences to trees / graphs
  - computation of reduct (intersection of a sentence/tree pair with a grammar)
  - expectation maximization (EM) training
  - automatic grammar refinement by the split/merge algorithm

Some of these concepts/algorithms have been described in the following articles:
- LCFRS/sDCP hybrid grammars and induction from discontinuous constituent structures 
    [[Nederhof/Vogler 2014]](https://www.aclweb.org/anthology/C14-1130)
- aligned hypergraph bimorphism, reducts and EM training [[Drewes/Gebhardt/Vogler 2016]](https://www.aclweb.org/anthology/W16-2407)
- general hybrid grammars, induction of LCFRS/sDCP hybrid grammars from discontinuous constituent trees and
    non-projective dependency trees [[Gebhardt/Nederhof/Vogler 2017]](https://doi.org/10.1162/COLI_a_00291)
- generic split/merge training, in partiuclar for LCFRS and LCFRS/sDCP hybrid grammars 
    [[Gebhardt (unpublished)]](no_yet)
    
Due to the purpose of the software (research), it is unstable and 
was tested on just 1-3 machines. Interfaces are likely to change. Maintenance 
is limited to Kilian's professional involvement in academic research.

------------------------------------------------------------------------

## Installation/ Preparation:

TODO: complete the description according to wiki at
<https://gitlab.tcs.inf.tu-dresden.de/hybrid-grammars/lcfrs-sdcp-hybrid-grammars/wikis/install-prerequesites>

1. Obtain python dependencies from pip by running

        pip3 setup -r [--user]

2. Install this patched version of `disco-dop` from branch [chart-exposure](https://github.com/kilian-gebhardt/disco-dop/tree/chart-exposure`).

3. Compile Cython modules by running

        python3 setup.py build_ext --inplace

------------------------------------------------------------------------

Resources / Corpora
-------------------

For many scripts it is assumed that various corpora are available in the res directory. You
may want to download or symlink them there:

`res/dependency_conll` -\> various corpora of CONLL-X shared task

`res/tiger/tiger_release_aug07.corrected.16012013.xml` -\> the tiger
corpus

`res/negra-corpus/downloadv2/negra-corpus.{cfg,export}` -\> the
negra-corpus

`res/wsj_dependency/{02-22,23,24}.conll` -\> various section of PTB/WSJ
in conll format

`res/negra-dep/negra-lower-punct-{train,test}.conll` -\> a conversion of negra
in conll format as described in [[Maier/Kallmeyer 2010]](http://www.wolfgang-maier.net/pub/tagplus10.pdf) 

TODO: complete the list

------------------------------------------------------------------------

## Unit tests

Running unit tests (requires some corpora to be available in the res
directory)

    python3 -m unittest discover

------------------------------------------------------------------------

## Documentation for the experiments in [[Gebhardt/Nederhof/Vogler 2017]](https://doi.org/10.1162/COLI_a_00291)

Actually an older version of this software was used to run the experiments in
this paper. Still, experiments can be reproduced as follows: 

- install the optional dependencies `grammatical framework` and `pynini`. 
  NB: It is possible to run the experiments without these requirements, however, there may be large differences in run times. The most probable parse might be ambiguous, i.e., 
      other parser implementations may select different parses.
- run the experiments as described below 

### Documentation for dependency parsing

Acquire a corpus in CoNLL-X shared task format, cf.
<http://ilk.uvt.nl/conll/post_task_data.html>

To run an experiment you have to create a configuration file of the
following format. With each line (that is not a comment) a parameter is
set. If one parameter is set multiple times, then the last value is
used. Each parameter where no default value is indicated needs to be
set.

    #                            This is a comment
    Database:                    path/to/experiment-db
    Training Corpus:             path/to/training/corpus
    Test Corpus:                 path/to/test/corpus
    Nonterminal Labeling:        child-cpos+deprel
    Terminal Labeling:           pos
    Recursive Partitioning:      left-branching
    Training Limit:              1000                   # default: unlimited
    Test Limit:                  200                    # default: unlimited
    Test Length Limit:           25                     # default: unlimited
    # Pre/Post-processing options
    Default Root DEPREL:         ROOT                   # default: do not overwrite
    Ignore Punctuation:          NO    # YES or NO      # default: NO
    Default Disconnected DEPREL: PUNC                   # default: _ (underscore)

------------------------------------------------------------------------

Implemented nonterminal labeling strategies:

    [strict|child|stricttop|childtop|empty]-[pos|cpos|deprel|cpos+deprel|pos+deprel]

Implemented terminal labeling strategies:

    [pos|cpos|form]

Implemented recursive partitioning strategies:

    [left-branching|right-branching|direct-extraction|cfg|fanout-$K]

where `$K > 0`

**Warning**: Ignoring punctuation is experimental and may raise errors if a punctuation symbol governs some non-punctuation word. Also, correct CoNLL export is not
    guaranteed in this case.

------------------------------------------------------------------------

Then run

    PYTHONPATH=. python3 experiment/cl_dependency_experiments.py path/to/configuration/file

This command will start the experiment. The program outputs various statistics on the grammar to stdout. Also, it writes the results in the experiment database. **Beware**: Also scores are printed, but they are calculated according to our own non-standard implementation. You may want to use the evaluation script described below which uses the standard CoNLL-X implementation. Also, in case of very short parse times, the run-time can be dominated by the database latency.

#### Evaluation

In order to list the contents of the experiment database run

    PYTHONPATH=. python3 evaluation/cl_dependency_evaluation.py path/to/database list

In order to generate a LaTeX file containing a table with various
statistics and scores run

    PYTHONPATH=. python3 evaluation/cl_dependency_evaluation.py path/to/database plot --experiments=$SELECTION --outfile=path/to/table.tex [--max-length=$N]

where `$SELECTION` is a list of natural numbers separated by `,` or
`-`, where each natural number references a row in the table and
`n-m` expands to `n,n+1, ..., m-1,m` and `n` needs to be
smaller than `m`. The order in the this list specifies the order in
the generated table. With `--max-length` a limit on the sentence
length can be specified, i.e., scores and parsing times will only
reflect sentences up to this length.

#### Dependency parsing, cascade experiment

In order to run the cascade experiment in [[Nederhof/Gebhardt/Vogler 2017, Table 3, p. 509]](),
run the following:
 
    PYTHONPATH=. python3 experiment/cl_dependency_cascade.py
   
No separate evaluation is required.

------------------------------------------------------------------------

### Documentation for Constituent Parsing

Acquire the Tiger and/or the Negra corpus.

In `corpora/tiger_parse.py`: Change the definitions of `TIGER_DIR` and
`TIGER`.

In `corpora/negra_parse.py`: Change the definition of `NEGRA_DIRECTORY`.

Uncomment the relevant lines of `experiment/cl_constituent_experiment.py` to select the desired
experiments.

Run: `PYTHONPATH=. python3 experiment/cl_constituent_experiments.py`


## Acknowledgements 
This project is an extension work by Mark-Jan Nederhof available [here](http://mjn.host.cs.st-andrews.ac.uk/code/coling2014.zip).

For parsing and preprocessing this project depends on the following libraries/packages: 
- [Grammatical Framework](https://www.grammaticalframework.org/)
- [OpenFST](http://www.openfst.org) / [pynini](http://www.openfst.org/twiki/bin/view/GRM/Pynini)
- [disco-dop](https://github.com/andreasvc/disco-dop/) by Andreas van Cranenburgh
- [treetools](https://github.com/wmaier/treetools) by Wolfgang Maier

The C++ Part of the project (S/M training) uses the [Boost](https://www.boost.org) and [Eigen](http://eigen.tuxfamily.org) libraries.

Evaluation scripts and parameter files are provided unter `./util`:
- `eval.pl` from the CoNLL-X shared task
- `proper.prm` taken from [disco-dop](https://github.com/andreasvc/disco-dop/) 