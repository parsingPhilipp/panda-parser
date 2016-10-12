//
// Created by kilian on 28/09/16.
//
#include "cfg.h"

using namespace cyk;

Rule::Rule(Nonterminal lhn, bool unary, int idx, double weight, bool left_term, bool right_term, unsigned left_child,
           unsigned right_child) :
    lhn(lhn), unary(unary), idx(idx), weight(weight), left_term(left_term), right_term(right_term), left_child(left_child),
right_child(right_child){}

Rule Rule::LexRule(Nonterminal lhn, int idx, double weight, unsigned terminal) {
    return Rule(lhn, true, idx, weight, true, false, terminal, 0);
}

Rule Rule::BinaryRule(Nonterminal lhn, int idx, double weight, unsigned left_child, unsigned right_child) {
    return Rule(lhn, false, idx, weight, false, false, left_child, right_child);
}

Rule Rule::ChainRule(Nonterminal lhn, int idx, double weight, unsigned nonterminal) {
    return Rule(lhn, true, idx, weight, false, false, nonterminal, 0);
}

void CFG::add_rule(Rule rule) {
    if (rule.lhn >= lhn_to_rules.size())
        lhn_to_rules.resize(rule.lhn + 1);
    lhn_to_rules[rule.lhn].push_back(rule);
    if (rule.unary && rule.left_term) {
        if (rule.left_child >= the_lex_rules.size())
            the_lex_rules.resize(rule.left_child + 1);
        the_lex_rules[rule.left_child].push_back(rule);
    }
    else {
        if (rule.left_child >= left_nont_corner.size())
            left_nont_corner.resize(rule.left_child + 1);
        left_nont_corner[rule.left_child].push_back(rule);
        if (! rule.unary) {
            if (rule.right_child >= right_nont_corner.size())
                right_nont_corner.resize(rule.right_child + 1);
            right_nont_corner[rule.right_child].push_back(rule);
        }
    }
}

CFG::CFG(Nonterminal initial) : initial(initial){}

CFG::CFG() : initial(0) {}

void CFG::set_initial(Nonterminal initial) {
    this->initial = initial;
}

void CFG::add_lex_rule(Nonterminal lhn, int idx, double weight, Terminal terminal) {
    this->add_rule(Rule::LexRule(lhn, idx, weight, terminal));
}

void CFG::add_chain_rule(Nonterminal lhn, int idx, double weight, Nonterminal rhs) {
    this->add_rule(Rule::ChainRule(lhn, idx, weight, rhs));
}

void CFG::add_binary_rule(Nonterminal lhn, int idx, double weight, Nonterminal left_child, Nonterminal right_child) {
    this->add_rule(Rule::BinaryRule(lhn, idx, weight, left_child, right_child));
}
