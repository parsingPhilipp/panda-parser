#! /bin/python3

from subprocess import call
import plac
import os

arg_str1 = "punctlower"
arg_str2 = "negrassplit,negravpsplit"


@plac.annotations(
    rparse=('path to rparse.jar', 'positional', None, str),
    negra=('path to negra-corpus.export', 'positional', None, str),
    result_dir=('path to result dir', 'positional', None, str)
    )
def main(rparse, negra, result_dir):
    for name, srange in zip(["train", "test"], ["1-18527", "18528-20602"]):
        dest = os.path.join(result_dir, "negra-lower-punct-" + name +".conll")
        dest_tmp = os.path.join(result_dir, "negra-corpus-" + name + ".export.tmp")
        call_str_1 = ["java", "-jar", rparse, "-doProcess",
                      "-outputFormat", "export", "-inputTreebank", negra,
                      "-outputTreebank", dest_tmp, "-inputEncoding", "ISO-8859-1", "-inputMaxlen", "25",
                      "-inputIntervals", srange, "-tasks", arg_str1]
        call_str_2 = ["java", "-jar",  rparse, "-doProcess",
                      "-outputFormat", "conll-hallnivrelabeleddep-negra", "-inputTreebank", dest_tmp,
                      "-outputTreebank", dest, "-inputMaxlen", "25",
                      "-tasks", arg_str2]
        # print(call_str)
        # print()
        call(call_str_1)
        call(call_str_2)


if __name__ == "__main__":
    plac.call(main)
