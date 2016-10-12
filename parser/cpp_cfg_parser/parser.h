//
// Created by kilian on 28/09/16.
//

#ifndef BINCFGPARSER_PARSER_H
#define BINCFGPARSER_PARSER_H

#include "cfg.h"
#include <unordered_map>
#include <map>
#include <memory>

namespace cyk {

    class CYKItem {
    public:
        Nonterminal nonterminal;
        unsigned left;
        unsigned right;
        double weight;
        CYKItem *left_child, *right_child;
        Rule *rule;

        int counter;

        CYKItem(Nonterminal nonterminal, unsigned left, unsigned right, Rule *rule);

        CYKItem(Nonterminal nonterminal, unsigned int left, unsigned int right, double weight, CYKItem *left_child,
                CYKItem *right_child, Rule *rule);
        ~CYKItem();
        bool operator<(const CYKItem otherCYKItem) const;

        int rule_idx() const;

    };

    class Range {
    public:
        unsigned left, right;
        bool operator==(const Range &otherRange) const;
        bool operator<(const Range otherRange) const;
        Range(unsigned int left, unsigned int right);
    };

    class Compare {
    public:
        bool operator()(std::shared_ptr<CYKItem>, std::shared_ptr<CYKItem>);
    };

}

namespace std {
    template<>
    struct hash<cyk::Range> {
        typedef cyk::Range argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const &r) const {
            result_type const h1(std::hash<unsigned>{}(r.left));
            result_type const h2(std::hash<unsigned>{}(r.right));
            return h1 ^ (h2 << 1); // or use boost::hash_combine
        }
    };
}

namespace cyk {
    class CYKParser {
    private:
        std::priority_queue<std::shared_ptr<CYKItem>, std::vector<std::shared_ptr<CYKItem>>, Compare> agenda;
        void do_parse();
        bool addToChart(std::shared_ptr<CYKItem> item);
        CYKItem * goal = nullptr;
        Terminal * input;
        unsigned length;
        std::vector<Rule> no_rules;

        std::vector<Rule> & lexical_rules(Terminal terminal) {
            if (grammar.the_lex_rules.size() > terminal)
                return grammar.the_lex_rules[terminal];
            else
                return no_rules;
        }

        std::vector<Rule> & nonterminal_left_corner(Nonterminal nonterminal) {
            if (grammar.left_nont_corner.size() > nonterminal)
                return grammar.left_nont_corner[nonterminal];
            else
                return no_rules;
        }

        std::vector<Rule> & nonterminal_right_corner(Nonterminal nonterminal) {
            if (grammar.right_nont_corner.size() > nonterminal)
                return grammar.right_nont_corner[nonterminal];
            else
                return no_rules;
        }
        void clear();

    public:

        CFG grammar;
        CYKParser();
        CYKParser(CFG grammar);

        std::vector<std::map<Range, std::shared_ptr<CYKItem>>> chart;

        void parse_input(Terminal *input, unsigned length);
        CYKItem* get_goal();

        ~CYKParser();

    };

    void print_derivation(CYKItem * root, int indent);
}


#endif //BINCFGPARSER_PARSER_H
