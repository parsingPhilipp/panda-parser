from parser.naive.parsing import LCFRS_parser as NaiveParser
from parser.fst.fst_export import RightBranchingFSTParser, LeftBranchingFSTParser
from parser.cpp_cfg_parser.parser_wrapper import CFGParser
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.parser_interface import AbstractParser
import re
from collections import defaultdict

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
            return GFParser
        return self.__parsers[name]


def the_parser_factory():
    factory = ParserFactory()
    # factory.registerParser('left-branching', LeftBranchingParser)
    # factory.registerParser('right-branching', RightBranchingParser)
    factory.registerParser('left-branching', LeftBranchingFSTParser)
    factory.registerParser('right-branching', RightBranchingFSTParser)
    factory.registerParser('direct-extraction', GFParser)
    factory.registerParser('naive-bottom-up', NaiveParser)
    factory.registerParser('fst-right-branching', RightBranchingFSTParser)
    factory.registerParser('fst-left-branching', LeftBranchingFSTParser)
    factory.registerParser('cfg-parser', CFGParser)
    factory.registerParser('gf-parser', GFParser)
    factory.registerParser('gf-parser-k-best', GFParser_k_best)
    return factory
