#ifndef BINCFGPARSER_CFG_H
#define BINCFGPARSER_CFG_H

#include <queue>

typedef unsigned Terminal;
typedef unsigned Nonterminal;

namespace cyk {

    class Rule {
    public:
        Nonterminal lhn;
        bool unary;
        int idx;
        double weight;
        bool left_term;
        bool right_term;
        unsigned left_child;
        unsigned right_child;

        Rule(Nonterminal lhn, bool unary, int idx, double weight, bool left_term, bool right_term, unsigned left_child,
             unsigned right_child);

        static Rule LexRule(Nonterminal lhn, int idx, double weight, unsigned terminal);

        static Rule ChainRule(Nonterminal lhn, int idx, double weight, unsigned nonterminal);

        static Rule BinaryRule(Nonterminal lhn, int idx, double weight, unsigned left_child, unsigned right_child);
    };


    class CFG {
    public:
        Nonterminal initial;

        void add_rule(Rule rule);
        void add_lex_rule(Nonterminal lhn, int idx, double weight, Terminal terminal);
        void add_chain_rule(Nonterminal lhn, int idx, double weight, Nonterminal rhs);
        void add_binary_rule(Nonterminal lhn, int idx, double weight, Nonterminal left_child, Nonterminal right_child);
        void set_initial(Nonterminal initial);

        std::vector<std::vector<Rule>> lhn_to_rules;
        std::vector<std::vector<Rule>> left_nont_corner;
        std::vector<std::vector<Rule>> right_nont_corner;
        std::vector<std::vector<Rule>> the_lex_rules;

        CFG();
        CFG(Nonterminal initial);
    };
}

#endif //BINCFGPARSER_CFG_H