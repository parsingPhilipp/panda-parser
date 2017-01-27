from grammar.lcfrs import LCFRS, LCFRS_var
from parser.LCFRS.LCFRS_Parser_Wrapper import PyLCFRSFactory



def parse_LCFRS(grammar, word):
    """
    :type grammar:LCFRS
    """
    factory = PyLCFRSFactory(grammar.start())
    for rule in grammar.rules(): # TODO: Get and transform rule IDs!
        factory.new_rule(rule.lhs().nont())
        for argument in rule.lhs().args():
            for symbol in argument:
                if type(symbol) is LCFRS_var:
                    factory.add_variable(symbol.mem, symbol.arg)
                else:
                    factory.add_terminal(symbol)
            factory.complete_argument()
        factory.add_rule_to_grammar(rule.rhs())

    factory.do_parse(word)
    print factory.get_passive_items_map()
    print "\n"
    print "\n"
    print factory.convert_trace()
    return

