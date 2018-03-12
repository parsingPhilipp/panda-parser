#! /bin/bash

# cat /tmp/tigerHN08-test.raw | java -jar ~/uni/implementation/corenlp/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar -annotators tokenize,ssplit -outputFormat conll 2>/dev/null > /tmp/tigerHN08-test.raw.conll
# cat -s /tmp/tigerHN08-test.raw.conll | awk -v OFS='\t' '{print $1,$2};' | less > /tmp/tigerHN08-test.raw.conll2


discodop treetransforms ../res/TIGER/tiger21/tigertest_root_attach.export /tmp/tigerHN08-test_root_moved.export --punct=move

treetools transform /tmp/tigerHN08-test_root_moved.export /tmp/tigerHN08-test.raw --dest-format terminals --dest-opts terminals

cat /tmp/tigerHN08-test.raw | ./raw2conll.py > /tmp/tigerHN08-test.raw.conll2

# mate-tools are required:
# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mate-tools/anna-3.61.jar
# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mate-tools/ger-tagger+lemmatizer+morphology+graph-based-3.6+.tgz

java -cp anna-3.61.jar is2.tag.Tagger -model tag-ger-3.6.model -test /tmp/tigerHN08-test.raw.conll2 -out /tmp/tigerHN08-test.tagged.conll

cat /tmp/tigerHN08-test.tagged.conll | ./join_lines.py > ../res/TIGER/tigerHN08-test.pred_tags.raw
