//
// Created by kilian on 28/09/16.
//

#include "parser.h"
#include <iostream>
#include <queue>

using namespace cyk;

int the_item_counter = 0;

CYKParser::CYKParser(CFG grammar) : grammar(grammar) {}

CYKItem::CYKItem(Nonterminal nont, unsigned int left, unsigned int right, Rule *rule) :
        CYKItem(nont, left, right, rule->weight, nullptr, nullptr, rule) {}

CYKItem::CYKItem(Nonterminal nonterminal, unsigned int left, unsigned int right, double weight, CYKItem *left_child,
                 CYKItem *right_child, Rule *rule) : nonterminal(nonterminal), left(left), right(right),
                                                     weight(weight), left_child(left_child),
                                                     right_child(right_child), rule(rule) {
    counter = ++the_item_counter;
}

bool CYKItem::operator<(const CYKItem otherCYKItem) const {
    return weight < otherCYKItem.weight;
}

CYKItem::~CYKItem() {
}

int CYKItem::rule_idx() const {
    return this->rule->idx;
}

bool CYKParser::addToChart(std::shared_ptr<CYKItem> item) {
    if (item->nonterminal >= chart.size()) {
        chart.resize(item->nonterminal + 1);
    }
    auto &nt_chart = chart[item->nonterminal];
    if (nt_chart.find(Range(item->left, item->right)) != nt_chart.end()) {
        return false;
    }
    nt_chart.emplace(Range(item->left, item->right), item);
    return true;
}

Range::Range(unsigned int left, unsigned int right) : left(left), right(right) {}

bool Range::operator==(const Range &otherRange) const {
    return (left == otherRange.left && right == otherRange.right);
}

bool Range::operator<(const Range otherRange) const {
    return (left < otherRange.left || left == otherRange.left && right < otherRange.right);
}

void CYKParser::do_parse() {
    std::shared_ptr<CYKItem> gen_item;
    for (unsigned i = 0; i < length; ++i) {
        for (Rule &rule : lexical_rules(input[i])) {
            gen_item = std::make_shared<CYKItem>(CYKItem(rule.lhn, i, i + 1, &rule));
            agenda.push(gen_item);
        }
    }

    std::vector<std::shared_ptr<CYKItem>> transport;
    std::shared_ptr<CYKItem> item_;

    while (!agenda.empty()) {
        item_ = agenda.top();
        CYKItem & item = *item_;
        agenda.pop();

        /*
        std::cout << "Process " << item.counter << " " << item.nonterminal
                  << ":" << item.left << "-" << item.right << " weight: " << item.weight;
        */

        if (!addToChart(item_)) {
            // std::cout << " skipped " << std::endl;
            continue;
        }

        // std::cout << " used" << std::endl;

        if (item.nonterminal == grammar.initial
            && item.left == 0
            && item.right == length) {
            goal = &item;
            break;
        }

        // item is left child
        //for (Rule &rule : grammar.left_nont_corner[item.nonterminal]) {
        for (Rule &rule : nonterminal_left_corner(item.nonterminal)) {
                // unary rules
            if (rule.unary) {
                gen_item = std::make_shared<CYKItem>(CYKItem(rule.lhn, item.left, item.right, item.weight + rule.weight, &item, nullptr, &rule));
                transport.push_back(gen_item);
            } else {
                if (rule.right_child < chart.size()) {
                    auto chart_range = chart[rule.right_child].lower_bound(Range(item.right, item.right));
                    while (chart_range != chart[rule.right_child].end()
                           && chart_range->first.left == item.right) {
                        gen_item = std::make_shared<CYKItem>(CYKItem(rule.lhn, item.left, chart_range->second->right,
                                              item.weight + rule.weight + chart_range->second->weight, &item,
                                              &(*(chart_range->second)), &rule));
                        transport.push_back(gen_item);
                        chart_range++;
                    }
                }
            }
        }

        // item is right child
        for (Rule &rule : nonterminal_right_corner(item.nonterminal)) {
            if (rule.left_child < chart.size()) {
                auto chart_range = chart[rule.left_child].begin();
                while (chart_range != chart[rule.left_child].end()
                       && chart_range->first.left <= item.left
                        && chart_range->first.right <= item.left) {
                    if (chart_range->first.right == item.left) {
                        gen_item = std::make_shared<CYKItem>(CYKItem(rule.lhn, chart_range->second->left, item.right,
                                               item.weight + rule.weight + chart_range->second->weight,
                                               &(*(chart_range->second)), &item, &rule));
                        transport.push_back(gen_item);
                    }
                    chart_range++;
                }
            }
        }

        // add new items to agenda if not present in chart
        for (std::shared_ptr<CYKItem> & new_item_ : transport) {
            CYKItem & new_item = *new_item_;
            if (new_item.nonterminal >= chart.size())
                chart.resize(new_item.nonterminal + 1);
            auto chart_range = chart[new_item.nonterminal].find(Range(new_item.left, new_item.right));
            if (chart_range == chart[new_item.nonterminal].end()) {
                agenda.push(new_item_);
            }
        }

        transport.clear();
    }
}

void CYKParser::parse_input(Terminal *input, unsigned length) {
    this->clear();
    this->input = input;
    this->length = length;
    do_parse();
}

CYKItem* CYKParser::get_goal() {
    return goal;
}

CYKParser::~CYKParser() {
    /*
    while(!agenda.empty()){
        CYKItem* top = agenda.top();
        agenda.pop();
        std::cout << "delete " << top->counter << " " << top << std::endl;
        delete top;
    }
    std::cout << "In chart: " << std::endl;
    for(auto i = 0; i < chart.size(); ++ i) {
        auto pointer = chart[i].begin();
        while (pointer != chart[i].end()){
            std::cout << "delete " << pointer->second->counter << " " << &(pointer->second) << std::endl;
            pointer ++;
        }

    }
     */
}

CYKParser::CYKParser() {}

void CYKParser::clear() {
    this->chart.clear();
    this->agenda = std::priority_queue<std::shared_ptr<CYKItem>, std::vector<std::shared_ptr<CYKItem>>, Compare>();
    this->goal = nullptr;
}

void ::cyk::print_derivation(CYKItem *root, int indent) {
    if (root) {
        for (auto i = 0; i < indent; i++)
            std::cout << "  ";
        std::cout << "rule " << root->rule_idx() << std::endl;
        print_derivation(root->left_child, indent + 1);
        print_derivation(root->right_child, indent + 1);
    }
}

bool Compare::operator()(std::shared_ptr<CYKItem> a, std::shared_ptr<CYKItem> b) {
    return *a < *b;
}
