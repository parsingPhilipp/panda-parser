from collections import defaultdict
import math as math
class Cluster:

    def __init__(self, cluster_id, init_word):
        self.cluster_id = cluster_id
        self.words = list()
        self.add_word(init_word)

    def add_word(self, word):
        self.words.append(word)

    def in_cluster(self, word):
        if word in self.words:
            return True
        else:
            return False

    def __repr__(self):
        return "Cluster_id: " + str(self.cluster_id) + " Words: " + str(self.words)


class BrownClustering:

    def __init__(self, in_file, num_clusters, out_file):

        self.corpus = [["ich", "habe", "heute", "geburtstag"], ["heute", "gab", "es", "grüne", "tomaten"], ["heute", "habe", "ich", "grüne", "tomaten", "gegessen"]]
        self.total_word_count = 0
        self.vocabulary = set()
        for sentence in self.corpus:
            self.total_word_count += len(sentence)
            for word in sentence:
                self.vocabulary.add(word)
        self.total_bigram_count = self.total_word_count - len(self.corpus)
        self.clusters = list()
        i = 0
        for word in self.vocabulary:
            self.clusters.append(Cluster(i, word))
            i += 1
        self.bigram_count = defaultdict(lambda: 0)
        self.as_prefix_count = defaultdict(lambda: 0)
        self.as_suffix_count = defaultdict(lambda: 0)
        self.non_zero_combination_prefix = defaultdict(lambda: set())
        self.non_zero_combination_suffix = defaultdict(lambda: set())
        for sentence in self.corpus:
            for i in range(len(sentence)-1):
                bigram = self.get_cluster_id(sentence[i]), self.get_cluster_id(sentence[i + 1])
                self.bigram_count[bigram] += 1
                self.as_prefix_count[bigram[0]] += 1
                self.as_suffix_count[bigram[1]] += 1
                self.non_zero_combination_prefix[bigram[0]].add(bigram[1])
                self.non_zero_combination_suffix[bigram[1]].add(bigram[0])

        self.q = defaultdict(lambda: 0)
        for tup in self.bigram_count.keys():
            count_tuple = self.bigram_count[tup]
            self.q[tup] = (count_tuple/self.total_bigram_count) * math.log10(
                (count_tuple*self.total_bigram_count) /
                (self.as_prefix_count[tup[0]]*self.as_suffix_count[tup[1]])
            )
        self.avg_mut_info = sum(self.q.values())
        self.s = defaultdict(lambda: 0)
        for cl in self.clusters:
            i = cl.cluster_id
            ql = 0
            qr = 0
            for cl_id in self.non_zero_combination_prefix[i]:
                ql += self.q[(i, cl_id)]
            for cl_id in self.non_zero_combination_suffix[i]:
                qr += self.q[(cl_id, i)]
            qs = 0
            if (i, i) in self.q.keys():
                qs = self.q[(i, i)]
            self.s[cl.cluster_id] = ql+qr-qs

        print("q values: " + str(self.q))
        print("average mutual information: " + str(self.avg_mut_info))
        print("count of bigram combinations: " + str(self.bigram_count))                  #pr(x,y) = bigram_count(x,y)/total_bigram_count
        print("count of key used as first entry of bigram: " + str(self.as_prefix_count)) #pl(x)= as_prefix_count(x)/total_bigram_count
        print("count of key used as second entry of bigram: " + str(self.as_suffix_count))#pr(x)= as_suffix_count(x)/total_bigram_count
        print("clusters: " + str(self.clusters))
        print("total word count: " + str(self.total_word_count))
        print("total bigram count: " + str(self.total_bigram_count))
        print("non zero combinations with key as first bigram entry: " + str(self.non_zero_combination_prefix)) #keep track of pr(x,y) couples
        print("non zero combinations with key as second bigram entry: " + str(self.non_zero_combination_suffix))
        print("s values: " + str(self.s))
        maximum_updated_avg_info = -1
        clusters_to_merge = (-1, -1)
        for cl_a in self.clusters:
            for cl_b in self.clusters:
                if cl_a.cluster_id < cl_b.cluster_id:
                    updated_avg_info = self.evaluate_merge(cl_a.cluster_id, cl_b.cluster_id)
                    if updated_avg_info > maximum_updated_avg_info:
                        maximum_updated_avg_info = updated_avg_info
                        clusters_to_merge = (cl_a.cluster_id, cl_b.cluster_id)
        print("Best merge couple is :" + str(clusters_to_merge) +
              " avg_mutual_info after merge: " + str(maximum_updated_avg_info))

    def get_cluster_id(self, word):
        for cl in self.clusters:
            if cl.in_cluster(word):
                return cl.cluster_id
        print("Error:Word " + word + " is in no cluster!")

    def evaluate_merge(self, cluster_id_a, cluster_id_b):
        """
        Calculates average mutual information after merging clusters with id cluster_id_a and cluster_id_b
        :param cluster_id_a:
        :param cluster_id_b:
        :return:
        """
        # http://www.aclweb.org/anthology/J92-4003 page 472 EQ.15
        # get all needed intermediate results
        # s_k(a), s_k(b)
        s_a = self.s[cluster_id_a]
        s_b = self.s[cluster_id_b]
        # q_k(a,b), q_k(b,a)
        q_ab = self.q[(cluster_id_a, cluster_id_b)]
        q_ba = self.q[(cluster_id_b, cluster_id_a)]

        # q_k(a+b,a+b)
        # 1.) calculate p_k(a+b,a+b) (not dividing by total_bigram_count here, always keeping counts dividing later)
        p_merged =\
            self.bigram_count[(cluster_id_a, cluster_id_a)] + self.bigram_count[(cluster_id_a, cluster_id_b)] +\
            self.bigram_count[(cluster_id_b, cluster_id_a)] + self.bigram_count[(cluster_id_b, cluster_id_b)]
        q_merged = 0
        # 2.) if p_k(a+b,a+b) == 0 q_k(a+b,a+b) will be 0 too
        if p_merged != 0:
            # calculate pl_k(a+b)*pr_k(a+b) -- pl_k(a,b) = pl_k(a) + pl_k(b)
            plr_merged = (self.as_prefix_count[cluster_id_a] + self.as_prefix_count[cluster_id_b]) *\
                         self.as_suffix_count[cluster_id_a] + self.as_suffix_count[cluster_id_b]
            # see EQ.14 -- now divide by total_bigram_count at needed locations to change form counts to probabilities
            q_merged = (p_merged/self.total_bigram_count) * math.log10((p_merged*self.total_bigram_count)/plr_merged)

        # sum(q_k(l,a+b)) l!=a,b
        # only consider combinations that occur in the corpus (where bigram_count > 0)
        # important copy! Requesting a set so need independent instance
        non_zero_suffix_combos = self.non_zero_combination_suffix[cluster_id_a].copy()
        non_zero_suffix_combos.update(self.non_zero_combination_suffix[cluster_id_b])
        # remove combinations where l in {a,b}
        if cluster_id_a in non_zero_suffix_combos:
            non_zero_suffix_combos.remove(cluster_id_a)
        if cluster_id_b in non_zero_suffix_combos:
            non_zero_suffix_combos.remove(cluster_id_b)
        q_merged_suffix = 0
        # for all occurring combinations calculate q(l,a+b)
        for prefix_id in non_zero_suffix_combos:
            # calculate p_k(l,a+b), pr_k(a+b), pl_k(l) then calculate q(l,a+b) as before (see EQ.14)
            p = self.bigram_count[(prefix_id, cluster_id_a)] + self.bigram_count[(prefix_id, cluster_id_b)]
            pr = self.as_suffix_count[cluster_id_a] + self.as_suffix_count[cluster_id_b]
            q_merged_suffix += (p / self.total_bigram_count) * math.log10(
                (p * self.total_bigram_count) / (pr * self.as_prefix_count[prefix_id]))

        # sum(q_k(a+b,l)) l!= a,b
        # analog to previous step
        non_zero_prefix_combos = self.non_zero_combination_prefix[cluster_id_a].copy()
        non_zero_prefix_combos.update(self.non_zero_combination_prefix[cluster_id_b])
        if cluster_id_a in non_zero_prefix_combos:
            non_zero_prefix_combos.remove(cluster_id_a)
        if cluster_id_b in non_zero_prefix_combos:
            non_zero_prefix_combos.remove(cluster_id_b)
        q_merged_prefix = 0
        for suffix_id in non_zero_prefix_combos:
            p = self.bigram_count[(cluster_id_a, suffix_id)] + self.bigram_count[(cluster_id_b, suffix_id)]
            pl = self.as_prefix_count[cluster_id_a] + self.as_prefix_count[cluster_id_b]
            q_merged_prefix += (p / self.total_bigram_count) * math.log10(
                (p * self.total_bigram_count) / (pl * self.as_suffix_count[suffix_id]))

        # combine as in EQ.15
        return self.avg_mut_info - s_a - s_b + q_ab + q_ba + q_merged + q_merged_suffix + q_merged_prefix

bc = BrownClustering("a", 2, "b")
