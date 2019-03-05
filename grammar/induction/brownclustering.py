from collections import defaultdict, OrderedDict
from copy import deepcopy
import math as math
import json
from os import path


class Cluster:

    def __init__(self, cluster_id, init_word):
        self.cluster_id = cluster_id
        self.words = list()
        self.words.append(init_word)

    def in_cluster(self, word):
        if word in self.words:
            return True
        else:
            return False

    def __repr__(self):
        return "Cluster_id: " + str(self.cluster_id) + " Words: " + str(self.words)


class BrownClustering:

    def __init__(self, corpus, num_clusters, out_file):
        self.desired_num_clusters = num_clusters
        self.corpus = corpus
        self.total_word_count = 0
        self.vocabulary = OrderedDict()
        for sentence in self.corpus:
            self.total_word_count += len(sentence)
            for word in sentence:
                self.vocabulary[word] = None
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

        self.word_bigram_count = defaultdict(lambda: 0)
        self.word_non_zero_combination_prefix = defaultdict(lambda: set())
        self.word_non_zero_combination_suffix = defaultdict(lambda: set())
        for sentence in self.corpus:
            for i in range(len(sentence)-1):
                bigram = self.get_cluster_id(sentence[i]), self.get_cluster_id(sentence[i + 1])
                self.bigram_count[bigram] += 1
                self.as_prefix_count[bigram[0]] += 1
                self.as_suffix_count[bigram[1]] += 1
                self.non_zero_combination_prefix[bigram[0]].add(bigram[1])
                self.non_zero_combination_suffix[bigram[1]].add(bigram[0])

        self.q = defaultdict(lambda: 0.0)
        for tup in self.bigram_count.keys():
            count_tuple = self.bigram_count[tup]
            self.q[tup] = (count_tuple/self.total_bigram_count) * math.log10(
                (count_tuple*self.total_bigram_count) /
                (self.as_prefix_count[tup[0]]*self.as_suffix_count[tup[1]])
            )
        self.avg_mut_info = sum(self.q.values())
        self.s = defaultdict(lambda: 0.0)
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

        #print("q values: " + str(self.q))
        #print("average mutual information: " + str(self.avg_mut_info))
        # p(x,y) = bigram_count(x,y)/total_bigram_count
        #print("count of bigram combinations: " + str(self.bigram_count))
        # pl(x)= as_prefix_count(x)/total_bigram_count
        #print("count of key used as first entry of bigram: " + str(self.as_prefix_count))
        # pr(x)= as_suffix_count(x)/total_bigram_count
        #print("count of key used as second entry of bigram: " + str(self.as_suffix_count))
        #print("clusters: " + str(self.clusters))
        #print("total word count: " + str(self.total_word_count))
        #print("total bigram count: " + str(self.total_bigram_count))
        # keep track of pr(x,y) couples
        #print("non zero combinations with key as first bigram entry: " + str(self.non_zero_combination_prefix))
        #print("non zero combinations with key as second bigram entry: " + str(self.non_zero_combination_suffix))
        #print("s values: " + str(self.s))
        self.avg_info_loss = defaultdict(lambda: 0.0)
        for cl_a in self.clusters:
            for cl_b in self.clusters:
                if cl_a.cluster_id < cl_b.cluster_id:
                    self.avg_info_loss[(cl_a.cluster_id, cl_b.cluster_id)] =\
                        self.evaluate_merge(cl_a.cluster_id, cl_b.cluster_id)
        clusters_to_merge = min(self.avg_info_loss, key=self.avg_info_loss.get)
        print("average mutual information loss by merging corresponding clusters: " + str(self.avg_info_loss))
        print("Best merge couple is :" + str(clusters_to_merge) +
              " avg_mutual_info loss by merge: " + str(self.avg_info_loss[clusters_to_merge]))

        self.merge_clusters(clusters_to_merge[0], clusters_to_merge[1])
        self.post_cluster_optimization()


        base_path = path.abspath(path.dirname(__file__))
        base_path = base_path[:-17]
        base_path += '/clustering/'
        with open(base_path + out_file+'.clustering', 'w', encoding='UTF-8') as out:
            json.dump(self.get_serialization(), out, ensure_ascii=False)

    def get_serialization(self):
        data = []
        for cl in self.clusters:
            data.append(cl.words)
        return data

    def get_cluster_id(self, word):
        for cl in self.clusters:
            if cl.in_cluster(word):
                return cl.cluster_id
        print("Error:Word " + word + " is in no cluster!")

    def get_cluster(self, cluster_id):
        for cl in self.clusters:
            if cl.cluster_id == cluster_id:
                return cl

    def evaluate_merge(self, cluster_id_a, cluster_id_b):
        """
        Calculates average mutual information loss after merging clusters with id cluster_id_a and cluster_id_b
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
        q_merged = self.calc_q(cl_id_l=cluster_id_a, cl_id_l2=cluster_id_b, cl_id_r=cluster_id_a, cl_id_r2=cluster_id_b)

        non_zero_prefix_combos = self.non_zero_combination_prefix[cluster_id_a].copy()
        non_zero_prefix_combos.update(self.non_zero_combination_prefix[cluster_id_b])
        if cluster_id_a in non_zero_prefix_combos:
            non_zero_prefix_combos.remove(cluster_id_a)
        if cluster_id_b in non_zero_prefix_combos:
            non_zero_prefix_combos.remove(cluster_id_b)

        q_merged_prefix = 0
        for cl_id in non_zero_prefix_combos:
            if cl_id != cluster_id_a and cl_id != cluster_id_b:
                q_merged_prefix += self.calc_q(cl_id_l=cluster_id_a, cl_id_l2=cluster_id_b, cl_id_r=cl_id)

        non_zero_suffix_combos = self.non_zero_combination_suffix[cluster_id_a].copy()
        non_zero_suffix_combos.update(self.non_zero_combination_suffix[cluster_id_b])
        # remove combinations where l in {a,b}
        if cluster_id_a in non_zero_suffix_combos:
            non_zero_suffix_combos.remove(cluster_id_a)
        if cluster_id_b in non_zero_suffix_combos:
            non_zero_suffix_combos.remove(cluster_id_b)

        q_merged_suffix = 0
        for cl_id in non_zero_suffix_combos:
            if cl_id != cluster_id_a and cl_id != cluster_id_b:
                q_merged_suffix += self.calc_q(cl_id_l=cl_id, cl_id_r=cluster_id_a, cl_id_r2=cluster_id_b)
        # combine as in EQ.15
        return s_a + s_b - q_ab - q_ba - q_merged - q_merged_suffix - q_merged_prefix

    def calc_q(self, cl_id_l, cl_id_r, cl_id_l2=None, cl_id_r2=None):

        p = self.bigram_count[(cl_id_l, cl_id_r)]
        pl = self.as_prefix_count[cl_id_l]
        pr = self.as_suffix_count[cl_id_r]
        if cl_id_l2 is not None:
            p += self.bigram_count[(cl_id_l2, cl_id_r)]
            pl += self.as_prefix_count[cl_id_l2]
        if cl_id_r2 is not None:
            pr += self.as_suffix_count[cl_id_r2]
            p += self.bigram_count[(cl_id_l, cl_id_r2)]
        if cl_id_r2 is not None and cl_id_l2 is not None:
            p += self.bigram_count[(cl_id_l2, cl_id_r2)]
        if p > 0:
            return (p/self.total_bigram_count)*math.log10((p*self.total_bigram_count)/(pl*pr))
        return 0

    def calc_q_temp(self, bigram_count, as_prefix_count, as_suffix_count, cl_id_l, cl_id_r, cl_id_l2=None,
                    cl_id_r2=None):

        p = bigram_count[(cl_id_l, cl_id_r)]
        pl = as_prefix_count[cl_id_l]
        pr = as_suffix_count[cl_id_r]
        if cl_id_l2 is not None:
            p += bigram_count[(cl_id_l2, cl_id_r)]
            pl += as_prefix_count[cl_id_l2]
        if cl_id_r2 is not None:
            pr += as_suffix_count[cl_id_r2]
            p += bigram_count[(cl_id_l, cl_id_r2)]
        if cl_id_r2 is not None and cl_id_l2 is not None:
            p += bigram_count[(cl_id_l2, cl_id_r2)]
        if p > 0:
            return (p / self.total_bigram_count) * math.log10((p * self.total_bigram_count) / (pl * pr))
        return 0

    def merge_clusters(self, cluster_id_a, cluster_id_b):

        while len(self.clusters) > self.desired_num_clusters:

            self.avg_mut_info -= self.avg_info_loss[(cluster_id_a, cluster_id_b)]
            # a,b --> a
            # update avg_info_loss, s by subtracting obsolete q values
            for cl_l in self.clusters:
                if cl_l.cluster_id != cluster_id_b and cl_l.cluster_id != cluster_id_a:
                    self.s[cl_l.cluster_id] -= self.q[(cl_l.cluster_id, cluster_id_a)]\
                                             + self.q[(cl_l.cluster_id, cluster_id_b)]\
                                             + self.q[(cluster_id_a, cl_l.cluster_id)]\
                                             + self.q[(cluster_id_b, cl_l.cluster_id)]
                    for cl_m in self.clusters:
                        if cl_l.cluster_id < cl_m.cluster_id != cluster_id_a and cl_m.cluster_id != cluster_id_b:
                            self.avg_info_loss[(cl_l.cluster_id, cl_m.cluster_id)] += \
                                self.calc_q(cl_id_l=cl_l.cluster_id, cl_id_l2=cl_m.cluster_id, cl_id_r=cluster_id_a)\
                                + self.calc_q(cl_id_l=cl_l.cluster_id, cl_id_l2=cl_m.cluster_id, cl_id_r=cluster_id_b)\
                                + self.calc_q(cl_id_l=cluster_id_a, cl_id_r=cl_l.cluster_id, cl_id_r2=cl_m.cluster_id)\
                                + self.calc_q(cl_id_l=cluster_id_b, cl_id_r=cl_l.cluster_id, cl_id_r2=cl_m.cluster_id) \
                                - self.q[(cl_l.cluster_id, cluster_id_a)]\
                                - self.q[(cl_m.cluster_id, cluster_id_a)]\
                                - self.q[(cluster_id_a, cl_l.cluster_id)]\
                                - self.q[(cluster_id_a, cl_m.cluster_id)]\
                                - self.q[(cl_l.cluster_id, cluster_id_b)]\
                                - self.q[(cl_m.cluster_id, cluster_id_b)]\
                                - self.q[(cluster_id_b, cl_l.cluster_id)]\
                                - self.q[(cluster_id_b, cl_m.cluster_id)]
            '''                
            # remove q values containing b
            for cl_id in self.non_zero_combination_prefix[cluster_id_b]:
                del self.q[(cluster_id_b, cl_id)]
            for cl_id in self.non_zero_combination_suffix[cluster_id_b]:
                # prevents deleting (b,b) tuple twice
                if cl_id != cluster_id_b:
                    del self.q[(cl_id, cluster_id_b)]
            '''
            # remove all obsolete values containing b
            # q(b,*/*,b), s(b) , avg_info_loss(b,*/*,b) cluster(b)
            for cl in self.clusters:
                if (cl.cluster_id, cluster_id_b) in self.q.keys():
                    del self.q[(cl.cluster_id, cluster_id_b)]
                if (cluster_id_b, cl.cluster_id) in self.q.keys():
                    del self.q[(cluster_id_b, cl.cluster_id)]

            self.s.pop(cluster_id_b)

            for cl in self.clusters:
                if cl.cluster_id != cluster_id_b:
                    if cluster_id_b < cl.cluster_id:
                        self.avg_info_loss.pop((cluster_id_b, cl.cluster_id))
                    else:
                        self.avg_info_loss.pop((cl.cluster_id, cluster_id_b))

            cla = self.get_cluster(cluster_id_a)
            clb = self.get_cluster(cluster_id_b)
            for w in clb.words:
                cla.words.append(w)
            self.clusters.remove(clb)

            # update bigram count and non_zero_combinations
            # update p where a/b are first entry p(a,*) -> p(a,*)+ p(b,*), p(b,*)->0
            for cl_id in self.non_zero_combination_prefix[cluster_id_b]:
                # make sure p(b,b) is added to p(a,a)
                if cl_id != cluster_id_b:
                    self.bigram_count[(cluster_id_a, cl_id)] += self.bigram_count.pop((cluster_id_b, cl_id))
                else:
                    self.bigram_count[(cluster_id_a, cluster_id_a)] += self.bigram_count.pop((cluster_id_b, cl_id))
            self.non_zero_combination_prefix[cluster_id_a].update(self.non_zero_combination_prefix.pop(cluster_id_b))
            # rename clusters in non_zero_combinations
            for non_zero_set in self.non_zero_combination_prefix.values():
                if cluster_id_b in non_zero_set:
                    non_zero_set.remove(cluster_id_b)
                    non_zero_set.add(cluster_id_a)
            # update p where a/b are second entry
            for cl_id in self.non_zero_combination_suffix[cluster_id_b]:
                # prevents double deleting/adding of bigram[(b,b)]
                if cl_id != cluster_id_b:
                    self.bigram_count[(cl_id, cluster_id_a)] += self.bigram_count.pop((cl_id, cluster_id_b))
            self.non_zero_combination_suffix[cluster_id_a].update(self.non_zero_combination_suffix.pop(cluster_id_b))
            # rename clusters in non_zero_combinations
            for non_zero_set in self.non_zero_combination_suffix.values():
                if cluster_id_b in non_zero_set:
                    non_zero_set.remove(cluster_id_b)
                    non_zero_set.add(cluster_id_a)
            # update pl/pr
            self.as_suffix_count[cluster_id_a] += self.as_suffix_count.pop(cluster_id_b)
            self.as_prefix_count[cluster_id_a] += self.as_prefix_count.pop(cluster_id_b)

            # recalculate q values containing merged a
            for cl_id in self.non_zero_combination_prefix[cluster_id_a]:
                self.q[(cluster_id_a, cl_id)] = self.calc_q(cl_id_l=cluster_id_a, cl_id_r=cl_id)
            for cl_id in self.non_zero_combination_suffix[cluster_id_a]:
                self.q[(cl_id, cluster_id_a)] = self.calc_q(cl_id_l=cl_id, cl_id_r=cluster_id_a)

            # finalize avg_info_loss
            # finalize s values

            for cl_l in self.clusters:
                if cl_l.cluster_id != cluster_id_a:
                    self.s[cl_l.cluster_id] += (self.q[(cl_l.cluster_id, cluster_id_a)]
                                                + self.q[(cluster_id_a, cl_l.cluster_id)])
                    for cl_m in self.clusters:
                        if cl_l.cluster_id < cl_m.cluster_id != cluster_id_a:
                            self.avg_info_loss[(cl_l.cluster_id, cl_m.cluster_id)] -= \
                                self.calc_q(cl_id_l=cl_l.cluster_id, cl_id_l2=cl_m.cluster_id, cl_id_r=cluster_id_a)\
                                + self.calc_q(cl_id_l=cluster_id_a, cl_id_r=cl_l.cluster_id, cl_id_r2=cl_m.cluster_id)\
                                - self.q[(cluster_id_a, cl_l.cluster_id)]\
                                - self.q[(cluster_id_a, cl_m.cluster_id)]\
                                - self.q[(cl_l.cluster_id, cluster_id_a)]\
                                - self.q[(cl_m.cluster_id, cluster_id_a)]
            # calculate s(a)
            ql = 0
            qr = 0
            for cl_id in self.non_zero_combination_prefix[cluster_id_a]:
                ql += self.q[(cluster_id_a, cl_id)]
            for cl_id in self.non_zero_combination_suffix[cluster_id_a]:
                qr += self.q[(cl_id, cluster_id_a)]
            qs = 0
            if (cluster_id_a, cluster_id_a) in self.q.keys():
                qs = self.q[(cluster_id_a, cluster_id_a)]
            self.s[cluster_id_a] = ql + qr - qs

            # calculate avg_info_loss(a,*/*,a)
            for cl in self.clusters:
                if cl.cluster_id != cluster_id_a:
                    if cluster_id_a < cl.cluster_id:
                        self.avg_info_loss[(cluster_id_a, cl.cluster_id)] =\
                            self.evaluate_merge(cluster_id_a, cl.cluster_id)
                    else:
                        self.avg_info_loss[(cl.cluster_id, cluster_id_a)] =\
                            self.evaluate_merge(cl.cluster_id, cluster_id_a)


            #print("#################################################")
            #print("count of bigram combinations: " + str(self.bigram_count))
            #print("count of key used as first entry of bigram: " + str(self.as_prefix_count))
            #print("count of key used as second entry of bigram: " + str(self.as_suffix_count))
            #print("non-zero-combinations with key as first bigram entry: " + str(self.non_zero_combination_prefix))
            #print("non-zero-combinations with key as second bigram entry: " + str(self.non_zero_combination_suffix))
            #print("q values: " + str(self.q))
            #print("s values: " + str(self.s))
            #print("avg_mutual_info_loss: " + str(self.avg_info_loss))
            #print("clusters: " + str(self.clusters))
            #print(sum(self.q.values()))
            #print(self.avg_mut_info)
            if len(self.clusters) > 1:
                best_merge_pair = min(self.avg_info_loss, key=self.avg_info_loss.get)
                print("next merge pair: " + str(best_merge_pair))
                cluster_id_a = best_merge_pair[0]
                cluster_id_b = best_merge_pair[1]

    def post_cluster_optimization(self):

        for sentence in self.corpus:
            for i in range(len(sentence)-1):
                self.word_bigram_count[(sentence[i], sentence[i+1])] += 1
                self.word_non_zero_combination_prefix[sentence[i]].add(sentence[i+1])
                self.word_non_zero_combination_suffix[sentence[i + 1]].add(sentence[i])

        has_changed = True
        while has_changed:
            has_changed = False
            for word in self.vocabulary:
                if self.move_to_best_cluster(word):
                    has_changed = True
        print(self.clusters)
        # TODO iterate over vocab, repeat if improvement was achieved

    def move_to_best_cluster(self, word):
        print('finding best cluster for: ' + word)
        print(self.bigram_count)
        print(self.non_zero_combination_prefix)
        print(self.non_zero_combination_suffix)
        print(self.as_prefix_count)
        print(self.as_suffix_count)
        print(self.q)
        # update values containing word
        # remove bigrams containing word
        print(self.word_bigram_count)
        print(self.clusters)
        print(self.word_non_zero_combination_prefix)
        print(self.word_non_zero_combination_suffix)
        print(self.bigram_count)

        affected_cluster_id = self.get_cluster_id(word)
        print(self.as_prefix_count[affected_cluster_id])
        print(self.as_suffix_count[affected_cluster_id])
        print(self.q)
        print(self.avg_mut_info)
        post_del_info = self.avg_mut_info
        for cl_id in self.non_zero_combination_prefix[affected_cluster_id]:
            post_del_info -= self.q.pop((affected_cluster_id, cl_id))
        for cl_id in self.non_zero_combination_suffix[affected_cluster_id]:
            if cl_id != affected_cluster_id:
                post_del_info -= self.q.pop((cl_id, affected_cluster_id))
        print(self.non_zero_combination_suffix)
        print(self.non_zero_combination_prefix)
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            obsolete_bigram = self.word_bigram_count[(word, suffix)]
            self.bigram_count[(affected_cluster_id, suffix_id)] -= obsolete_bigram
            self.as_prefix_count[affected_cluster_id] -= obsolete_bigram
            if self.bigram_count[(affected_cluster_id, self.get_cluster_id(suffix))] == 0:
                self.non_zero_combination_prefix[affected_cluster_id].remove(suffix_id)
                self.non_zero_combination_suffix[suffix_id].remove(affected_cluster_id)
        for prefix in self.word_non_zero_combination_suffix[word]:
            # prevent double subtraction of 'word','word' bigrams
            prefix_id = self.get_cluster_id(prefix)
            obsolete_bigram = self.word_bigram_count[(prefix, word)]
            if prefix != word:
                self.bigram_count[(prefix_id, affected_cluster_id)] -= obsolete_bigram
                if self.bigram_count[(prefix_id, affected_cluster_id)] == 0:
                    self.non_zero_combination_prefix[prefix_id].remove(affected_cluster_id)
                    self.non_zero_combination_suffix[affected_cluster_id].remove(prefix_id)
            self.as_suffix_count[affected_cluster_id] -= obsolete_bigram
        print(self.q)
        for cl_id in self.non_zero_combination_prefix[affected_cluster_id]:
            updated_q_value = self.calc_q(cl_id_l=affected_cluster_id, cl_id_r=cl_id)
            self.q[(affected_cluster_id, cl_id)] = updated_q_value
            post_del_info += updated_q_value
        for cl_id in self.non_zero_combination_suffix[affected_cluster_id]:
            if cl_id != affected_cluster_id:
                updated_q_value = self.calc_q(cl_id_l=cl_id, cl_id_r=affected_cluster_id)
                self.q[(cl_id, affected_cluster_id)] = updated_q_value
                post_del_info += updated_q_value
        print(self.non_zero_combination_suffix)
        print(self.non_zero_combination_prefix)
        print(self.bigram_count)
        print(self.as_prefix_count)
        print(self.as_suffix_count)
        print(self.q)
        print(post_del_info)
        move_improvement = dict()
        for cl in self.clusters:
            move_improvement[cl.cluster_id] = self.evaluate_move(word=word, cluster_id=cl.cluster_id, post_del_info=post_del_info) - self.avg_mut_info
        print(move_improvement)
        if max(move_improvement.values()) > 0.00000000001:
            self.perform_move(word=word, cluster_id=max(move_improvement, key=move_improvement.get), post_del_info=post_del_info)
            return True
        else:
            self.perform_move(word=word, cluster_id=affected_cluster_id, post_del_info=post_del_info)
            return False
        # TODO actually move to cluster with biggest improvement

    def evaluate_move(self, word, cluster_id, post_del_info):
        # move word into new cluster
        self.get_cluster(self.get_cluster_id(word)).words.remove(word)
        self.get_cluster(cluster_id).words.append(word)
        bigram_count = self.bigram_count.copy()
        non_zero_combination_prefix = deepcopy(self.non_zero_combination_prefix)
        non_zero_combination_suffix = deepcopy(self.non_zero_combination_suffix)
        as_prefix_count = self.as_prefix_count.copy()
        as_suffix_count = self.as_suffix_count.copy()
        for cl_id in non_zero_combination_prefix[cluster_id]:
            post_del_info -= self.q[(cluster_id, cl_id)]
        for cl_id in non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                post_del_info -= self.q[(cl_id, cluster_id)]
        print(bigram_count)
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            new_bigram = self.word_bigram_count[(word, suffix)]
            bigram_count[(cluster_id, suffix_id)] += new_bigram
            as_prefix_count[cluster_id] += new_bigram
            non_zero_combination_prefix[cluster_id].add(suffix_id)
            non_zero_combination_suffix[suffix].add(cluster_id)
        for prefix in self.word_non_zero_combination_suffix[word]:
            prefix_id = self.get_cluster_id(prefix)
            new_bigram = self.word_bigram_count[(prefix, word)]
            if prefix != word:
                bigram_count[(prefix_id, cluster_id)] += new_bigram
            as_suffix_count[cluster_id] += new_bigram
            non_zero_combination_suffix[cluster_id].add(prefix_id)
            non_zero_combination_prefix[prefix_id].add(cluster_id)
        print(bigram_count)
        print(non_zero_combination_prefix)
        print(non_zero_combination_suffix)
        print(as_prefix_count)
        print(as_suffix_count)
        for cl_id in non_zero_combination_prefix[cluster_id]:
            updated_q_value = self.calc_q_temp(bigram_count=bigram_count, as_prefix_count=as_prefix_count, as_suffix_count=as_suffix_count, cl_id_l=cluster_id, cl_id_r=cl_id)
            post_del_info += updated_q_value
        for cl_id in non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                updated_q_value = self.calc_q_temp(bigram_count=bigram_count, as_prefix_count=as_prefix_count, as_suffix_count=as_suffix_count, cl_id_l=cl_id, cl_id_r=cluster_id)
                post_del_info += updated_q_value
        print(post_del_info)
        return post_del_info

    def perform_move(self, word, cluster_id, post_del_info):
        # move word into new cluster
        print(self.non_zero_combination_prefix)
        self.get_cluster(self.get_cluster_id(word)).words.remove(word)
        self.get_cluster(cluster_id).words.append(word)
        for cl_id in self.non_zero_combination_prefix[cluster_id]:
            post_del_info -= self.q.pop((cluster_id, cl_id))
        for cl_id in self.non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                post_del_info -= self.q.pop((cl_id, cluster_id))
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            new_bigram = self.word_bigram_count[(word, suffix)]
            self.bigram_count[(cluster_id, suffix_id)] += new_bigram
            self.as_prefix_count[cluster_id] += new_bigram
            self.non_zero_combination_prefix[cluster_id].add(suffix_id)
            self.non_zero_combination_suffix[suffix_id].add(cluster_id)
        for prefix in self.word_non_zero_combination_suffix[word]:
            prefix_id = self.get_cluster_id(prefix)
            new_bigram = self.word_bigram_count[(prefix, word)]
            if prefix != word:
                self.bigram_count[(prefix_id, cluster_id)] += new_bigram
            self.as_suffix_count[cluster_id] += new_bigram
            self.non_zero_combination_suffix[cluster_id].add(prefix_id)
            self.non_zero_combination_prefix[prefix_id].add(cluster_id)
        for cl_id in self.non_zero_combination_prefix[cluster_id]:
            updated_q_value = self.calc_q(cl_id_l=cluster_id, cl_id_r=cl_id)
            self.q[(cluster_id, cl_id)] = updated_q_value
            post_del_info += updated_q_value
        for cl_id in self.non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                updated_q_value = self.calc_q(cl_id_l=cl_id, cl_id_r=cluster_id)
                self.q[(cl_id, cluster_id)] = updated_q_value
                post_del_info += updated_q_value
        self.avg_mut_info = post_del_info
        print('##################################')
        print('moved ' + word + ' to cluster: ' + str(cluster_id))
        print(self.bigram_count)
        print(self.non_zero_combination_prefix)
        print(self.non_zero_combination_suffix)
        print(self.as_prefix_count)
        print(self.as_suffix_count)
        print(self.q)
        print(sum(self.q.values()))
        print(self.avg_mut_info)
