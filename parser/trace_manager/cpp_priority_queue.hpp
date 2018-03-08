// cpp_priority_queue.hpp
#include <functional>
#include <queue>

template<typename T>
using comperator_function = std::function<bool(Element<T>&,Element<T>&)>;

template <typename T>
using intp_std_func_priority_queue = std::priority_queue<Element<T>,std::vector<Element<T>>,comperator_function<T>>;


template<typename T>
comperator_function<T> construct_comparator(std::map<Element<T>, double> & weights) {
    return [&weights](Element<T>& elem1, Element<T>& elem2) -> bool {
        return weights.at(elem1) < weights.at(elem2);
    };
}