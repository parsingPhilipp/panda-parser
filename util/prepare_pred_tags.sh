#! /bin/bash

discodop treetransforms ../res/TIGER/tiger21/tigertest_root_attach.export /tmp/tigerHN08-test_root_moved.export --punct=move

treetools transform /tmp/tigerHN08-test_root_moved.export /tmp/tigerHN08-test.raw --dest-format terminals --dest-opts terminals

cat /tmp/tigerHN08-test.raw | ./raw2conll.py > /tmp/tigerHN08-test.raw.conll2

# mate-tools are required:
# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mate-tools/anna-3.61.jar
# used tiger2.2 in conll9 format for training
# filter training data using filter_conll.py script, i.e., ./filter_conll.py tiger_release_aug07.corrected.16012013.conll09.conll > ../res/TIGER/tigerHN08-train+dev-split.release2.2.conll09
# train using: java -Xmx2G -cp anna-3.61.jar is2.tag.Tagger -model models/tag-v1-de-train+dev.model -train ../res/TIGER/tigerHN08-train+dev-split.release2.2.conll09

java -cp anna-3.61.jar is2.tag.Tagger -model tag-ger-3.6.model -test /tmp/tigerHN08-test.raw.conll2 -out /tmp/tigerHN08-test.tagged.conll

cat /tmp/tigerHN08-test.tagged.conll | ./join_lines.py > ../res/TIGER/tigerHN08-test.pred_tags.raw