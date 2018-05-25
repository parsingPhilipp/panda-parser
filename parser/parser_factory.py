import re
from collections import defaultdict
from parser.naive.parsing import LCFRS_parser as NaiveParser
from parser.cpp_cfg_parser.parser_wrapper import CFGParser


class ParserFactory:
    def __init__(self):
        self.__parsers = defaultdict(lambda: NaiveParser)

    def registerParser(self, name, parser):
        self.__parsers[name] = parser

    def getParser(self, name):
        # if name == "fanout-1":
        #    return self.__parsers["cfg-parser"]
        match = re.search(r'fanout-(\d+)', name)
        if match:
            # return ViterbiParser
            if 'gf-parser' in self.__parsers:
                return self.__parsers['gf-parser']
            elif match.group(1) == '1':
                return self.__parsers['cfg-parser']
            else:
                return self.__parsers['naive-bottom-up']
        return self.__parsers[name]


def the_parser_factory():
    factory = ParserFactory()
    # factory.registerParser('left-branching', LeftBranchingParser)
    # factory.registerParser('right-branching', RightBranchingParser)
    try:
        from parser.fst.fst_export import RightBranchingFSTParser, LeftBranchingFSTParser
        factory.registerParser('right-branching', RightBranchingFSTParser)
        factory.registerParser('fst-right-branching', RightBranchingFSTParser)
        factory.registerParser('left-branching', LeftBranchingFSTParser)
        factory.registerParser('fst-left-branching', LeftBranchingFSTParser)
    except ImportError as e:
        pass

    factory.registerParser('naive-bottom-up', NaiveParser)
    factory.registerParser('cfg-parser', CFGParser)

    try:
        from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
        factory.registerParser('direct-extraction', GFParser)
        factory.registerParser('gf-parser', GFParser)
        factory.registerParser('gf-parser-k-best', GFParser_k_best)
    except ImportError as e:
        pass

    return factory


__all__ = ["ParserFactory", "the_parser_factory"]
