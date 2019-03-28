# cython: profile=True
from collections import defaultdict, OrderedDict
from copy import deepcopy
import math as math
import json
import numpy as np
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

    def set_cluster_id(self,new_id):
        self.cluster_id=new_id

    def __repr__(self):
        return "Cluster_id: " + str(self.cluster_id) + " Words: " + str(self.words)

def getMinAndIndex(lst):
    cdef float minval
    cdef int minind
    minval,minidx = lst[0],0
    for i,v in enumerate(lst[1:]):
        if v < minval:
            minval,minidx = v,i+1
    return minval,minidx

def getMinIndex(lst):
    subMins = [getMinAndIndex(sub) for sub in lst ]
    subMin,subIdx = getMinAndIndex( [s[0] for s in subMins ] )

    return((subIdx, subMins[subIdx][1]))

class BrownClustering:

    def __init__(self, corpus, num_clusters, out_file, max_vocab_size=1000, optimization=True):
        print("init brown clustering")
        # Setup phase

        self.out_file = out_file
        self.desired_num_clusters = num_clusters
        self.corpus = corpus
        self.total_word_count = 0
        self.vocabulary = OrderedDict()
        word_queue = defaultdict(lambda :0)
        for sentence in self.corpus:
            self.total_word_count += len(sentence)
            for word in sentence:
                self.vocabulary[word] = None
                word_queue[word] += 1
        self.word_queue = list(word_queue.items())
        self.word_queue.sort(key= lambda x:x[1], reverse=True)
        #print(self.word_queue)
        self.total_bigram_count = self.total_word_count - len(self.corpus)
        #self.total_bigram_count = 0
        # initializing Clusters
        self.clusters = list()
        max_vocab_size = min(max_vocab_size, len(self.vocabulary))
        for i in range(max_vocab_size):
            self.clusters.append(Cluster(i, self.word_queue[i][0]))
        print("vocabulary size: " + str(len(self.vocabulary)))
        self.max_vocab_size = max_vocab_size
        self.word_bigram_count = defaultdict(lambda: 0)
        self.word_non_zero_combination_prefix = defaultdict(lambda: set())
        self.word_non_zero_combination_suffix = defaultdict(lambda: set())
        # create cluster based counts
        self.bigram_count = np.zeros([max_vocab_size, max_vocab_size], dtype=int).tolist()
        self.as_prefix_count = np.zeros(max_vocab_size, dtype=int).tolist()
        self.as_suffix_count = np.zeros(max_vocab_size, dtype=int).tolist()
        self.non_zero_combination_prefix = defaultdict(lambda: set())
        self.non_zero_combination_suffix = defaultdict(lambda: set())
        # create word based counts needed for post optimization

        # setup word based counts
        for sentence in self.corpus:
            for i in range(len(sentence)-1):
                self.word_bigram_count[(sentence[i],sentence[i+1])] += 1
                self.word_non_zero_combination_prefix[sentence[i]].add(sentence[i+1])
                self.word_non_zero_combination_suffix[sentence[i + 1]].add(sentence[i])
                #print(sentence[i]+sentence[i+1])
                bigram = (self.get_cluster_id(sentence[i]), self.get_cluster_id(sentence[i + 1]))
                #print(bigram)
                if bigram[0] != -1 and bigram[1] != -1:
                    #self.total_bigram_count += 1
                    self.bigram_count[bigram[0]][bigram[1]] += 1
                    self.as_prefix_count[bigram[0]] += 1
                    self.as_suffix_count[bigram[1]] += 1
                    self.non_zero_combination_prefix[bigram[0]].add(bigram[1])
                    self.non_zero_combination_suffix[bigram[1]].add(bigram[0])

        self.next_word_index = max_vocab_size

        '''
        # initialize cluster based counts
        for sentence in self.corpus:
            for i in range(len(sentence)-1):
                bigram = (self.get_cluster_id(sentence[i]), self.get_cluster_id(sentence[i + 1]))
                self.bigram_count[bigram[0]][bigram[1]] += 1
                self.as_prefix_count[bigram[0]] += 1
                self.as_suffix_count[bigram[1]] += 1
                self.non_zero_combination_prefix[bigram[0]].add(bigram[1])
                self.non_zero_combination_suffix[bigram[1]].add(bigram[0])
        '''
        # calculate q, s and avg_mut_information of initial clusters
        #self.q = defaultdict(lambda: 0.0)
        self.q = np.zeros([max_vocab_size, max_vocab_size], dtype=float).tolist()
        #for tup in self.bigram_count.keys():
        for prefix_comb in self.non_zero_combination_prefix.keys():
            for suffix in self.non_zero_combination_prefix[prefix_comb]:
                tup = (prefix_comb, suffix)
                count_tuple = self.bigram_count[tup[0]][tup[1]]
                self.q[tup[0]][tup[1]] = (count_tuple/self.total_bigram_count) * math.log(
                    (count_tuple*self.total_bigram_count) /
                    (self.as_prefix_count[tup[0]]*self.as_suffix_count[tup[1]])
                )
        self.avg_mut_info = sum(map(sum, self.q))
        #self.s = defaultdict(lambda: 0.0)
        self.s = np.zeros(len(self.vocabulary), dtype=float).tolist()
        for cl in self.clusters:
            i = cl.cluster_id
            ql = 0
            qr = 0
            for cl_id in self.non_zero_combination_prefix[i]:
                ql += self.q[i][ cl_id]
            for cl_id in self.non_zero_combination_suffix[i]:
                qr += self.q[cl_id][ i]
            #qs = 0
            #if (i, i) in self.q.keys():
            qs = self.q[i][i]
            self.s[cl.cluster_id] = ql+qr-qs
        # evaluate information loss for first merge
        #self.avg_info_loss = defaultdict(lambda: 0.0)
        inf_loss = np.ones([max_vocab_size, max_vocab_size], dtype=float)* 1000000
        self.avg_info_loss = inf_loss.tolist()
        for cl_a in self.clusters:
            for cl_b in self.clusters:
                if cl_a.cluster_id < cl_b.cluster_id:
                    self.avg_info_loss[cl_a.cluster_id][cl_b.cluster_id] =\
                        self.evaluate_merge(cl_a.cluster_id, cl_b.cluster_id)
        # get best pair for initial merge
        print("Starting merging process..")
        #clusters_to_merge = min(self.avg_info_loss, key=self.avg_info_loss.get)
        clusters_to_merge = getMinIndex(self.avg_info_loss)
        #clusters_to_merge = np.unravel_index(np.argmin(self.avg_info_loss, axis=None), self.avg_info_loss.shape)
        # start merging process with best pair
        # keep splitting until corpus is reduced to desired number of clusters
        self.merge_clusters(clusters_to_merge[0], clusters_to_merge[1])
        # if optimization after greedy clustering is desired
        # for each word in the vocab try to find a different cluster, where avg_mut_info increases
        # if a word was moved during this process - repeat until no more words are moved
        self.save_clustering(out_file=self.out_file+'_pre_optimization')
        print("Initial clustering completed!")
        if optimization:
            print("Starting post optimization process..")
            self.post_cluster_optimization()
            print("Optimization completed!")
            self.save_clustering(out_file=self.out_file+'_final')
        # save resulting clustering to file

        '''
        base_path = path.abspath(path.dirname(__file__))
        base_path = base_path[:-17]
        base_path += '/clustering/'
        print("Saving clustering in " + base_path + out_file+".clustering")
        with open(base_path + out_file+'.clustering', 'w', encoding='UTF-8') as out:
            json.dump(self.get_serialization(), out, ensure_ascii=False)
        '''


    def save_clustering(self, out_file):
        base_path = path.abspath(path.dirname(__file__))
        base_path = base_path[:-17]
        base_path += '/clustering/'
        print("Saving clustering in " + base_path + out_file + ".clustering")
        with open(base_path + out_file + '.clustering', 'w', encoding='UTF-8') as out:
            json.dump(self.get_serialization(), out, ensure_ascii=False)

    def get_serialization(self):
        """
        Turns Cluster list into serializable nested array
        :return: nested cluster array
        """
        data = []
        for cl in self.clusters:
            data.append(cl.words)
        return data

    def get_cluster_id(self, word):
        """
        Helper function to get the corresponding cluster_id of a word
        :param word:
        :return: cluster_id
        """
        for cl in self.clusters:
            if cl.in_cluster(word):
                return cl.cluster_id
        return -1

    def get_cluster(self, cluster_id):
        """
        Helper function to get the cluster corresponding to the cluster_id
        :param cluster_id:
        :return: cluster
        """
        for cl in self.clusters:
            if cl.cluster_id == cluster_id:
                return cl

    def add_next_word(self,cluster_id):
        '''
        print("before merge:")
        print("clusters: "+ str(self.clusters))
        print("total_bigram_count " + str(self.total_bigram_count))
        print("bigram_count: " + str(self.bigram_count))
        print("as_prefix_count: " +str(self.as_prefix_count))
        print("as_suffix_count: "+ str(self.as_suffix_count))
        print("non_zero_combination_prefix: " +str(self.non_zero_combination_prefix))
        print("non_zero_combination_suffix: " + str(self.non_zero_combination_suffix))
        print("q_values: " +str(self.q))
        print("s_values: " +str(self.s))
        print("avg_info_loss: " +str(self.avg_info_loss))
        print("initializing " + str(self.word_queue[self.next_word_index][0]) + " as cluster " + str(cluster_id))
        '''
        self.clusters.append(Cluster(cluster_id,self.word_queue[self.next_word_index][0]))

        for cl in self.clusters:
            for word in cl.words:
                cl_id  = self.get_cluster_id(word)
                count = self.word_bigram_count[(self.word_queue[self.next_word_index][0],word)]
                if count > 0:
                    self.bigram_count[cluster_id][cl_id] += count
                    self.as_prefix_count[cluster_id] += count
                    self.as_suffix_count[cl_id] += count
                    self.non_zero_combination_prefix[cluster_id].add(cl_id)
                    self.non_zero_combination_suffix[cl_id].add(cluster_id)
                # prevent double add of cluster_id,cluster_id
                if cl_id != cluster_id:
                    count = self.word_bigram_count[word,(self.word_queue[self.next_word_index][0])]
                    if count > 0:
                        self.bigram_count[cl_id][cluster_id] += count
                        self.as_prefix_count[cl_id] += count
                        self.as_suffix_count[cluster_id] += count
                        self.non_zero_combination_prefix[cl_id].add(cluster_id)
                        self.non_zero_combination_suffix[cluster_id].add(cl_id)
        '''
        for word_bigram in self.word_bigram_count:
            cluster_bigram = (self.get_cluster_id(word_bigram[0]),self.get_cluster_id(word_bigram[1]))
            if (cluster_bigram[0] != -1 and cluster_bigram[1] == cluster_id) or (cluster_bigram[0] == cluster_id and cluster_bigram[1]!=-1):
                count = self.word_bigram_count[word_bigram]
                self.bigram_count[cluster_bigram[0]][cluster_bigram[1]] += count
                ##self.total_bigram_count += count
                self.as_prefix_count[cluster_bigram[0]] += count
                self.as_suffix_count[cluster_bigram[1]] += count
                self.non_zero_combination_prefix[cluster_bigram[0]].add(cluster_bigram[1])
                self.non_zero_combination_suffix[cluster_bigram[1]].add(cluster_bigram[0])
        '''
        ql = 0
        for suffix in self.non_zero_combination_prefix[cluster_id]:
            q_val = self.calc_q_basic(cluster_id,suffix)
            self.q[cluster_id][suffix] = q_val
            ql +=  q_val
        qr  = 0
        for prefix in self.non_zero_combination_suffix[cluster_id]:
            q_val = self.calc_q_basic(prefix,cluster_id)
            self.q[prefix][cluster_id] = q_val
            qr += q_val
        self.s[cluster_id] = ql+qr-self.calc_q_basic(cluster_id,cluster_id)
        for cl in self.clusters:
            if cluster_id != cl.cluster_id:
                if cl.cluster_id < cluster_id:
                    self.avg_info_loss[cl.cluster_id][cluster_id] =  self.evaluate_merge(cl.cluster_id,cluster_id)
                else:
                    self.avg_info_loss[cluster_id][cl.cluster_id] = self.evaluate_merge(cluster_id,cl.cluster_id)

        self.next_word_index += 1
        '''
        print("after adding:")
        print("clusters: "+ str(self.clusters))
        print("total_bigram_count " + str(self.total_bigram_count))
        print("bigram_count: " + str(self.bigram_count))
        print("as_prefix_count: " +str(self.as_prefix_count))
        print("as_suffix_count: "+ str(self.as_suffix_count))
        print("non_zero_combination_prefix: " +str(self.non_zero_combination_prefix))
        print("non_zero_combination_suffix: " + str(self.non_zero_combination_suffix))
        print("q_values: " +str(self.q))
        print("s_values: " +str(self.s))
        print("avg_info_loss: " +str(self.avg_info_loss))
        '''
    def evaluate_merge(self, cluster_id_a, cluster_id_b):
        """
        Calculates average mutual information loss suffered by merging clusters with id cluster_id_a and cluster_id_b
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
        q_ab = self.q[cluster_id_a][ cluster_id_b]
        q_ba = self.q[cluster_id_b][ cluster_id_a]
        # q_k(a+b,a+b)
        q_merged = self.calc_q_full(cl_id_l=cluster_id_a, cl_id_l2=cluster_id_b, cl_id_r=cluster_id_a, cl_id_r2=cluster_id_b)
        # sum(q_k(a+b,l)) l!=a,b
        non_zero_prefix_combos = self.non_zero_combination_prefix[cluster_id_a].copy()
        non_zero_prefix_combos.update(self.non_zero_combination_prefix[cluster_id_b])
        # remove combinations where l in {a,b}
        if cluster_id_a in non_zero_prefix_combos:
            non_zero_prefix_combos.remove(cluster_id_a)
        if cluster_id_b in non_zero_prefix_combos:
            non_zero_prefix_combos.remove(cluster_id_b)

        q_merged_prefix = 0
        for cl_id in non_zero_prefix_combos:
            if cl_id != cluster_id_a and cl_id != cluster_id_b:
                q_merged_prefix += self.calc_q_l2(cl_id_l=cluster_id_a, cl_id_l2=cluster_id_b, cl_id_r=cl_id)

        # sum(q_k(l,a+b)) l!=a,b
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
                q_merged_suffix += self.calc_q_r2(cl_id_l=cl_id, cl_id_r=cluster_id_a, cl_id_r2=cluster_id_b)
        # combine as in EQ.15
        return s_a + s_b - q_ab - q_ba - q_merged - q_merged_suffix - q_merged_prefix

    def calc_q_basic(self,int cl_id_l,int cl_id_r):
        cdef int p,pl,pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0

    def calc_q_l2(self, int cl_id_l, int cl_id_l2, int cl_id_r):
        cdef int p, pl, pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        p += self.bigram_count[cl_id_l2][ cl_id_r]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            pl += self.as_prefix_count[cl_id_l2]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0

    def calc_q_r2(self, int cl_id_l, int cl_id_r, int cl_id_r2):
        cdef int p,pl,pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        p += self.bigram_count[cl_id_l][ cl_id_r2]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            pr += self.as_suffix_count[cl_id_r2]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0

    def calc_q_full(self,int cl_id_l,int cl_id_l2,int cl_id_r,int cl_id_r2):
        cdef int p,pl,pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        p += self.bigram_count[cl_id_l][ cl_id_r2]
        p += self.bigram_count[cl_id_l2][ cl_id_r]
        p += self.bigram_count[cl_id_l2][ cl_id_r2]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            pl += self.as_prefix_count[cl_id_l2]
            pr += self.as_suffix_count[cl_id_r2]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0

    def calc_q_temp(self, bigram_count, as_prefix_count, as_suffix_count, cl_id_l, cl_id_r):
        cdef int p,pl,pr
        p = bigram_count[cl_id_l][ cl_id_r]
        if p > 0:
            pl = as_prefix_count[cl_id_l]
            pr = as_suffix_count[cl_id_r]
            return (p / self.total_bigram_count) * math.log((p * self.total_bigram_count) / (pl * pr))
        return 0

    def merge_clusters(self, cluster_id_a, cluster_id_b):
        """
        Creates initial greedy clustering, starts by merging cluster_a with cluster_b
        :param cluster_id_a:
        :param cluster_id_b:
        :return:
        """
        print("first merge pair:" + str((cluster_id_a, cluster_id_b)))
        while len(self.clusters) > self.desired_num_clusters:

            self.avg_mut_info -= self.avg_info_loss[cluster_id_a][ cluster_id_b]
            # a,b --> a
            # update avg_info_loss, s by subtracting obsolete q values
            for cl_l in self.clusters:
                if cl_l.cluster_id != cluster_id_b and cl_l.cluster_id != cluster_id_a:
                    self.s[cl_l.cluster_id] -= self.q[cl_l.cluster_id][ cluster_id_a]\
                                             + self.q[cl_l.cluster_id][ cluster_id_b]\
                                             + self.q[cluster_id_a][ cl_l.cluster_id]\
                                             + self.q[cluster_id_b][ cl_l.cluster_id]
                    # warning eq.17 given in paper is incorrect! Use this modified version
                    for cl_m in self.clusters:
                        if cl_l.cluster_id < cl_m.cluster_id != cluster_id_a and cl_m.cluster_id != cluster_id_b:
                            self.avg_info_loss[cl_l.cluster_id][ cl_m.cluster_id] += \
                                self.calc_q_l2(cl_id_l=cl_l.cluster_id, cl_id_l2=cl_m.cluster_id, cl_id_r=cluster_id_a)\
                                + self.calc_q_l2(cl_id_l=cl_l.cluster_id, cl_id_l2=cl_m.cluster_id, cl_id_r=cluster_id_b)\
                                + self.calc_q_r2(cl_id_l=cluster_id_a, cl_id_r=cl_l.cluster_id, cl_id_r2=cl_m.cluster_id)\
                                + self.calc_q_r2(cl_id_l=cluster_id_b, cl_id_r=cl_l.cluster_id, cl_id_r2=cl_m.cluster_id) \
                                - self.q[cl_l.cluster_id][ cluster_id_a]\
                                - self.q[cl_m.cluster_id][ cluster_id_a]\
                                - self.q[cluster_id_a][ cl_l.cluster_id]\
                                - self.q[cluster_id_a][ cl_m.cluster_id]\
                                - self.q[cl_l.cluster_id][ cluster_id_b]\
                                - self.q[cl_m.cluster_id][ cluster_id_b]\
                                - self.q[cluster_id_b][ cl_l.cluster_id]\
                                - self.q[cluster_id_b][ cl_m.cluster_id]
            # remove all obsolete values containing b
            # q(b,*/*,b), s(b) , avg_info_loss(b,*/*,b) cluster(b)
            for cl in self.clusters:
                #if (cl.cluster_id, cluster_id_b) in self.q.keys():
                self.q[cl.cluster_id][ cluster_id_b] = 0
                #if (cluster_id_b, cl.cluster_id) in self.q.keys():
                self.q[cluster_id_b][ cl.cluster_id] = 0
            # remove s values of b
            self.s[cluster_id_b] = 0
            # remove avg_info_loss estimates containing b
            for cl in self.clusters:
                if cluster_id_b < cl.cluster_id:
                    self.avg_info_loss[cluster_id_b][ cl.cluster_id] = 1000000
                else:
                    self.avg_info_loss[cl.cluster_id][ cluster_id_b] = 1000000
            # move words of cluster b into cluster a then remove cluster b
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
                    self.bigram_count[cluster_id_a][ cl_id] += self.bigram_count[cluster_id_b][ cl_id]
                    self.bigram_count[cluster_id_b][ cl_id] = 0
                else:
                    self.bigram_count[cluster_id_a][ cluster_id_a] += self.bigram_count[cluster_id_b][ cl_id]
                    self.bigram_count[cluster_id_b][ cl_id] = 0
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
                    self.bigram_count[cl_id][ cluster_id_a] += self.bigram_count[cl_id][ cluster_id_b]
                    self.bigram_count[cl_id][ cluster_id_b] = 0
            self.non_zero_combination_suffix[cluster_id_a].update(self.non_zero_combination_suffix.pop(cluster_id_b))
            # rename clusters in non_zero_combinations
            for non_zero_set in self.non_zero_combination_suffix.values():
                if cluster_id_b in non_zero_set:
                    non_zero_set.remove(cluster_id_b)
                    non_zero_set.add(cluster_id_a)
            # update pl/pr
            self.as_suffix_count[cluster_id_a] += self.as_suffix_count[cluster_id_b]
            self.as_suffix_count[cluster_id_b] = 0
            self.as_prefix_count[cluster_id_a] += self.as_prefix_count[cluster_id_b]
            self.as_prefix_count[cluster_id_b] = 0
            # recalculate q values containing merged a
            for cl_id in self.non_zero_combination_prefix[cluster_id_a]:
                self.q[cluster_id_a][ cl_id] = self.calc_q_basic(cl_id_l=cluster_id_a, cl_id_r=cl_id)
            for cl_id in self.non_zero_combination_suffix[cluster_id_a]:
                self.q[cl_id][ cluster_id_a] = self.calc_q_basic(cl_id_l=cl_id, cl_id_r=cluster_id_a)

            # finalize avg_info_loss
            # finalize s values

            for cl_l in self.clusters:
                if cl_l.cluster_id != cluster_id_a:
                    self.s[cl_l.cluster_id] += (self.q[cl_l.cluster_id][ cluster_id_a]
                                                + self.q[cluster_id_a][cl_l.cluster_id])
                    # once again be aware that the eq.17 given in the paper is incorrect!
                    for cl_m in self.clusters:
                        if cl_l.cluster_id < cl_m.cluster_id != cluster_id_a:
                            self.avg_info_loss[cl_l.cluster_id][ cl_m.cluster_id] -= \
                                self.calc_q_l2(cl_id_l=cl_l.cluster_id, cl_id_l2=cl_m.cluster_id, cl_id_r=cluster_id_a)\
                                + self.calc_q_r2(cl_id_l=cluster_id_a, cl_id_r=cl_l.cluster_id, cl_id_r2=cl_m.cluster_id)\
                                - self.q[cluster_id_a][ cl_l.cluster_id]\
                                - self.q[cluster_id_a][ cl_m.cluster_id]\
                                - self.q[cl_l.cluster_id][ cluster_id_a]\
                                - self.q[cl_m.cluster_id][ cluster_id_a]
            # calculate s(a)
            ql = 0
            qr = 0
            for cl_id in self.non_zero_combination_prefix[cluster_id_a]:
                ql += self.q[cluster_id_a][ cl_id]
            for cl_id in self.non_zero_combination_suffix[cluster_id_a]:
                qr += self.q[cl_id][ cluster_id_a]
            #qs = 0
            #if (cluster_id_a, cluster_id_a) in self.q.keys():
            qs = self.q[cluster_id_a][ cluster_id_a]
            self.s[cluster_id_a] = ql + qr - qs

            # calculate avg_info_loss(a,*/*,a)
            for cl in self.clusters:
                if cl.cluster_id != cluster_id_a:
                    if cluster_id_a < cl.cluster_id:
                        self.avg_info_loss[cluster_id_a][ cl.cluster_id] =\
                            self.evaluate_merge(cluster_id_a, cl.cluster_id)
                    else:
                        self.avg_info_loss[cl.cluster_id][ cluster_id_a] =\
                            self.evaluate_merge(cl.cluster_id, cluster_id_a)

            # print("#################################################")
            # print("count of bigram combinations: " + str(self.bigram_count))
            # print("count of key used as first entry of bigram: " + str(self.as_prefix_count))
            # print("count of key used as second entry of bigram: " + str(self.as_suffix_count))
            # print("non-zero-combinations with key as first bigram entry: " + str(self.non_zero_combination_prefix))
            # print("non-zero-combinations with key as second bigram entry: " + str(self.non_zero_combination_suffix))
            # print("q values: " + str(self.q))
            # print("s values: " + str(self.s))
            # print("avg_mutual_info_loss: " + str(self.avg_info_loss))
            # print("clusters: " + str(self.clusters))
            # print(sum(self.q.values()))
            # print(self.avg_mut_info)
            if self.next_word_index < len(self.word_queue):
                self.add_next_word(cluster_id_b)

            # set clusters for next merge
            if len(self.clusters) > 1:
                #best_merge_pair = min(self.avg_info_loss, key=self.avg_info_loss.get)
                best_merge_pair = getMinIndex(self.avg_info_loss)
                #best_merge_pair = np.unravel_index(np.argmin(self.avg_info_loss, axis=None), self.avg_info_loss.shape)
                print("next merge pair: " + str(best_merge_pair))
                cluster_id_a = best_merge_pair[0]
                cluster_id_b = best_merge_pair[1]


    def cluster_reordering(self):
        i = 0
        mapping = dict()
        new_q = np.zeros([len(self.clusters),len(self.clusters)],dtype=float).tolist()
        new_bigram_count = np.zeros([len(self.clusters),len(self.clusters)],dtype=int).tolist()
        for cluster in self.clusters:
            mapping[cluster.cluster_id]=i
            cluster.set_cluster_id(i)
            i = i+1
        #print(mapping)
        for i in range(self.max_vocab_size):
            for j in range(self.max_vocab_size):
                if self.bigram_count[i][j] != 0:
                    new_bigram_count[mapping[i]][mapping[j]] = self.bigram_count[i][j]
                if self.q[i][j] != 0:
                    new_q[mapping[i]][mapping[j]] = self.q[i][j]

        new_as_prefix_count = np.zeros(len(self.clusters), dtype=int).tolist()
        new_as_suffix_count = np.zeros(len(self.clusters), dtype=int).tolist()
        for i in range(self.max_vocab_size):
            if self.as_prefix_count[i] != 0:
                new_as_prefix_count[mapping[i]] = self.as_prefix_count[i]
            if self.as_suffix_count[i] != 0:
                new_as_suffix_count[mapping[i]] = self.as_suffix_count[i]
        new_non_zero_combination_prefix = defaultdict(lambda: set())
        new_non_zero_combination_suffix = defaultdict(lambda: set())
        for key in self.non_zero_combination_prefix.keys():
            old_set = self.non_zero_combination_prefix[key]
            new_set = set()
            for val in old_set:
                new_set.add(mapping[val])
            new_non_zero_combination_prefix[mapping[key]] = new_set
        for key in self.non_zero_combination_suffix.keys():
            old_set = self.non_zero_combination_suffix[key]
            new_set = set()
            for val in old_set:
                new_set.add(mapping[val])
            new_non_zero_combination_suffix[mapping[key]] = new_set
        self.q = new_q
        self.bigram_count = new_bigram_count
        self.as_prefix_count = new_as_prefix_count
        self.as_suffix_count = new_as_suffix_count
        self.non_zero_combination_prefix = new_non_zero_combination_prefix
        self.non_zero_combination_suffix = new_non_zero_combination_suffix


    def post_cluster_optimization(self):
        """
        Performs optimization on initial clustering, by trying to find a better cluster for each word
        until no further improvement is found
        :return:
        """
        self.cluster_reordering()
        # checks whether words were moved in the last round
        has_changed = True
        round_counter = 0
        while has_changed:
            round_counter +=1
            has_changed = False
            for word in self.vocabulary:
                if self.move_to_best_cluster(word):
                    has_changed = True
            if has_changed:
                self.save_clustering(out_file=self.out_file + '_opt_round_' + str(round_counter))
        print(self.clusters)

    def move_to_best_cluster(self, word):
        """
        attempts to move a word to a better cluster
        :param word:
        :return: has_moved
        """
        # first remove the word out of its current cluster and update all counts accordingly
        # set cluster_id that currently contains word
        affected_cluster_id = self.get_cluster_id(word)
        post_del_info = self.avg_mut_info
        # remove q values containing the affected cluster_id
        # these will later be recalculated using updated bigram counts
        # keep track of avg_mut_info
        for cl_id in self.non_zero_combination_prefix[affected_cluster_id]:
            post_del_info -= self.q[affected_cluster_id][ cl_id]
            self.q[affected_cluster_id][ cl_id] = 0
        for cl_id in self.non_zero_combination_suffix[affected_cluster_id]:
            if cl_id != affected_cluster_id:
                post_del_info -= self.q[cl_id][affected_cluster_id]
                self.q[cl_id][ affected_cluster_id] = 0
        # look at bigrams containing word and remove them form the bigram counts
        # update the as_prefix/as_suffix count of the affected cluster
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            obsolete_bigram = self.word_bigram_count[(word, suffix)]
            self.bigram_count[affected_cluster_id][ suffix_id] -= obsolete_bigram
            self.as_prefix_count[affected_cluster_id] -= obsolete_bigram
            if self.bigram_count[affected_cluster_id][ self.get_cluster_id(suffix)] == 0:
                self.non_zero_combination_prefix[affected_cluster_id].remove(suffix_id)
                self.non_zero_combination_suffix[suffix_id].remove(affected_cluster_id)
        for prefix in self.word_non_zero_combination_suffix[word]:
            # prevent double subtraction of 'word','word' bigrams
            prefix_id = self.get_cluster_id(prefix)
            obsolete_bigram = self.word_bigram_count[(prefix, word)]
            if prefix != word:
                self.bigram_count[prefix_id][ affected_cluster_id] -= obsolete_bigram
                if self.bigram_count[prefix_id][ affected_cluster_id] == 0:
                    self.non_zero_combination_prefix[prefix_id].remove(affected_cluster_id)
                    self.non_zero_combination_suffix[affected_cluster_id].remove(prefix_id)
            # make sure the as suffix count is updated here even if concerning 'word''word' bigrams
            # as we only updated the prefix counts before
            self.as_suffix_count[affected_cluster_id] -= obsolete_bigram
        # recalculate the q values that we have removed earlier using the updated counts
        # keep track of the avg_mut_info
        for cl_id in self.non_zero_combination_prefix[affected_cluster_id]:
            updated_q_value = self.calc_q_basic(cl_id_l=affected_cluster_id, cl_id_r=cl_id)
            self.q[affected_cluster_id][ cl_id] = updated_q_value
            post_del_info += updated_q_value
        for cl_id in self.non_zero_combination_suffix[affected_cluster_id]:
            # beware of (affected_cluster, affected_cluster) value so the avg_mut_info is not increased twice
            if cl_id != affected_cluster_id:
                updated_q_value = self.calc_q_basic(cl_id_l=cl_id, cl_id_r=affected_cluster_id)
                self.q[cl_id][ affected_cluster_id] = updated_q_value
                post_del_info += updated_q_value
        # create list for the resulting avg_mut_info change be moving 'word' to different clusters
        move_improvement = dict()
        # evaluate possible moves
        for cl in self.clusters:
            move_improvement[cl.cluster_id] = self.evaluate_move(word=word, cluster_id=cl.cluster_id,
                                                                 post_del_info=post_del_info) - self.avg_mut_info
        # if there is a cluster that results in an increase of mut_info move word to this cluster
        # otherwise move it back to its original cluster
        # threshold used to negate possible improvements by floating point errors
        if max(move_improvement.values()) > 0.00000000001:
            self.perform_move(word=word, cluster_id=max(move_improvement, key=move_improvement.get),
                              post_del_info=post_del_info)
            return True
        else:
            self.perform_move(word=word, cluster_id=affected_cluster_id, post_del_info=post_del_info)
            return False

    def evaluate_move(self, word, cluster_id, post_del_info):
        """
        calculates the avg_mut_info after moving word into new cluster
        :param word: word that is evaluated
        :param cluster_id: cluster_id of cluster that is evaluated
        :param post_del_info: avg_mut_info after removing word
        :return: avg_mut_info
        """
        # move word into new cluster
        self.get_cluster(self.get_cluster_id(word)).words.remove(word)
        self.get_cluster(cluster_id).words.append(word)
        # use copy of counts so we do not mess with out current counts
        bigram_count = deepcopy(self.bigram_count)
        non_zero_combination_prefix = deepcopy(self.non_zero_combination_prefix)
        non_zero_combination_suffix = deepcopy(self.non_zero_combination_suffix)
        as_prefix_count = deepcopy(self.as_prefix_count)
        as_suffix_count = deepcopy(self.as_suffix_count)
        # remove q values using the cluster from avg_mut_info we want to move to, these will later be recalculated
        for cl_id in non_zero_combination_prefix[cluster_id]:
            post_del_info -= self.q[cluster_id][ cl_id]
        for cl_id in non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                post_del_info -= self.q[cl_id][ cluster_id]
        # add bigrams containing word to bigram counts corresponding to the new cluster
        # first for bigrams were 'word' is the prefix
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            new_bigram = self.word_bigram_count[(word,suffix)]
            bigram_count[cluster_id][ suffix_id] += new_bigram
            # increase prefix count of cluster by prefix count of 'word'
            as_prefix_count[cluster_id] += new_bigram
            # make sure non_zero_combinations are update with new combinations
            non_zero_combination_prefix[cluster_id].add(suffix_id)
            non_zero_combination_suffix[suffix].add(cluster_id)
        # now for bigrams were 'word' is the suffix
        for prefix in self.word_non_zero_combination_suffix[word]:
            prefix_id = self.get_cluster_id(prefix)
            new_bigram = self.word_bigram_count[(prefix, word)]
            # prevent double adding of 'word','word' bigrams
            if prefix != word:
                bigram_count[prefix_id][ cluster_id] += new_bigram
            # increase suffix count
            as_suffix_count[cluster_id] += new_bigram
            # add new combinations
            non_zero_combination_suffix[cluster_id].add(prefix_id)
            non_zero_combination_prefix[prefix_id].add(cluster_id)
        # add updated q values
        for cl_id in non_zero_combination_prefix[cluster_id]:
            updated_q_value = self.calc_q_temp(bigram_count=bigram_count, as_prefix_count=as_prefix_count,
                                               as_suffix_count=as_suffix_count, cl_id_l=cluster_id, cl_id_r=cl_id)
            post_del_info += updated_q_value
        for cl_id in non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                updated_q_value = self.calc_q_temp(bigram_count=bigram_count, as_prefix_count=as_prefix_count,
                                                   as_suffix_count=as_suffix_count, cl_id_l=cl_id, cl_id_r=cluster_id)
                post_del_info += updated_q_value
        return post_del_info

    def perform_move(self, word, cluster_id, post_del_info):
        """
        Moves a word to a new cluster, similar as evaluate_move put retains performed changes
        :param word:
        :param cluster_id:
        :param post_del_info:
        :return:
        """
        # this functions works very similar to evaluate cluster but works on the global counts
        # move word into new cluster
        self.get_cluster(self.get_cluster_id(word)).words.remove(word)
        self.get_cluster(cluster_id).words.append(word)
        # delete q values corresponding to new cluster and update avg_mut_info
        for cl_id in self.non_zero_combination_prefix[cluster_id]:
            post_del_info -= self.q[cluster_id][ cl_id]
            self.q[cluster_id][ cl_id] = 0
        for cl_id in self.non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                post_del_info -= self.q[cl_id][ cluster_id]
                self.q[cl_id][ cluster_id] = 0
        # update bigram, prefix and suffix counts and add new combinations
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            new_bigram = self.word_bigram_count[(word,suffix)]
            self.bigram_count[cluster_id][ suffix_id] += new_bigram
            self.as_prefix_count[cluster_id] += new_bigram
            self.non_zero_combination_prefix[cluster_id].add(suffix_id)
            self.non_zero_combination_suffix[suffix_id].add(cluster_id)
        for prefix in self.word_non_zero_combination_suffix[word]:
            prefix_id = self.get_cluster_id(prefix)
            new_bigram = self.word_bigram_count[(prefix, word)]
            if prefix != word:
                self.bigram_count[prefix_id][cluster_id] += new_bigram
            self.as_suffix_count[cluster_id] += new_bigram
            self.non_zero_combination_suffix[cluster_id].add(prefix_id)
            self.non_zero_combination_prefix[prefix_id].add(cluster_id)
        # recalculate q values corresponding to new cluster and update avg_mut_info
        for cl_id in self.non_zero_combination_prefix[cluster_id]:
            updated_q_value = self.calc_q_basic(cl_id_l=cluster_id, cl_id_r=cl_id)
            self.q[cluster_id][ cl_id] = updated_q_value
            post_del_info += updated_q_value
        for cl_id in self.non_zero_combination_suffix[cluster_id]:
            if cl_id != cluster_id:
                updated_q_value = self.calc_q_basic(cl_id_l=cl_id, cl_id_r=cluster_id)
                self.q[cl_id][ cluster_id] = updated_q_value
                post_del_info += updated_q_value
        if post_del_info > self.avg_mut_info + 0.00000000001:
            print('moved ' + word + ' to cluster: ' + str(cluster_id))
        self.avg_mut_info = post_del_info


