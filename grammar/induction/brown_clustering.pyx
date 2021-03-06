#cython: language_level=3
from collections import defaultdict
import math as math
import json
import numpy as np
from os import path


class Cluster:
    """
    Cluster objects containing the list of words related to this cluster.
    Each word should only belong to one cluster/cluster_id
    """
    def __init__(self, cluster_id, init_word):
        self.cluster_id = cluster_id
        self.words = list()
        self.words.append(init_word)


    def in_cluster(self, word):
        """
        helper functions checks whether word belongs to this cluster
        :param: word
        :return: True if word is in cluster
        """
        if word in self.words:
            return True
        else:
            return False

    def set_cluster_id(self,new_id):
        """
        setter to rename the clusters id, used to reorder clusters after initial clustering
        :param new_id: the clusters id is changed to this parameter
        :return:
        """
        self.cluster_id=new_id

    def __repr__(self):
        return "Cluster_id: " + str(self.cluster_id) + " Words: " + str(self.words)


def read_corpus_file(filename):
    base_path = path.abspath(path.dirname(__file__))
    base_path = base_path[:-17]
    base_path += 'res/'
    text_file = open(base_path+filename+'.raw')
    lines = text_file.read().splitlines()
    text_file.close()
    corpus = []
    for x in range(len(lines)):
        line = str.split(lines[x])
        lower_line = [x.lower() for x in line]
        corpus.append(lower_line)
    return corpus

class BrownClustering:
    """
    Creates a clustering using the given corpus input, following the brown cluster methodology (Brown et al. 1992)
    Attributes:
        :param: corpus: Input corpus: the algorithm expects a list of lists containing tokenized sentences
        :param: num_clusters: the number of clusters the input is reduced to
        :param: out_file: base name for the resulting json files, Files are save to the /clustering folder
        :param: max_vocab_size: limits the number words/clusters that are considered at any given time.
        New words are introduced, after space has been freed up by merging. WARNING: lowering the value of this
        parameter will drastically improve clustering speed but will yield less accurate clusterings. This parameter
        MUST be chosen larger then num_clusters
        :param: optimization: enables clustering optimization after the initial greedy clustering, by moving single
        words to better fitting clusters
    """

    def __init__(self, corpus_file, num_clusters, out_file, max_vocab_size=1000, optimization=True):
        print("init brown clustering")
        # Setup phase
        self.out_file = out_file
        self.desired_num_clusters = num_clusters
        self.corpus = read_corpus_file(corpus_file)
        self.total_word_count = 0
        self.vocabulary = set()
        # create order in which words are introduced to the clustering
        word_queue = defaultdict(lambda :0)
        for sentence in self.corpus:
            self.total_word_count += len(sentence)
            for word in sentence:
                self.vocabulary.add(word)
                word_queue[word] += 1
        self.word_queue = list(word_queue.items())
        self.word_queue.sort(key= lambda x:x[1], reverse=True)
        # total_bigram_count is calculated in full even if not all bigrams are considered in the clustering at the
        # beginning. The subsequent increase of this count would make all pre-calculated q values inaccurate
        self.total_bigram_count = self.total_word_count - len(self.corpus)
        print(self.total_bigram_count)
        # initializing Clusters
        self.clusters = list()
        max_vocab_size = min(max_vocab_size, len(self.vocabulary))
        for i in range(max_vocab_size):
            self.clusters.append(Cluster(i, self.word_queue[i][0]))
        print("vocabulary size: " + str(len(self.vocabulary)))
        self.max_vocab_size = max_vocab_size
        # initialize word based counts
        self.word_bigram_count = defaultdict(lambda: 0)
        self.word_non_zero_combination_prefix = defaultdict(lambda: set())
        self.word_non_zero_combination_suffix = defaultdict(lambda: set())
        # initialize cluster based counts
        self.bigram_count = np.zeros([max_vocab_size, max_vocab_size], dtype=int).tolist()
        self.as_prefix_count = np.zeros(max_vocab_size, dtype=int).tolist()
        self.as_suffix_count = np.zeros(max_vocab_size, dtype=int).tolist()
        self.non_zero_combination_prefix = defaultdict(lambda: set())
        self.non_zero_combination_suffix = defaultdict(lambda: set())


        # setup word based counts
        for sentence in self.corpus:
            for i in range(len(sentence)-1):
                self.word_bigram_count[(sentence[i],sentence[i+1])] += 1
                self.word_non_zero_combination_prefix[sentence[i]].add(sentence[i+1])
                self.word_non_zero_combination_suffix[sentence[i + 1]].add(sentence[i])
                bigram = (self.get_cluster_id(sentence[i]), self.get_cluster_id(sentence[i + 1]))
                # set up cluster based counts
                if bigram[0] != -1 and bigram[1] != -1:
                    self.bigram_count[bigram[0]][bigram[1]] += 1
                    self.as_prefix_count[bigram[0]] += 1
                    self.as_suffix_count[bigram[1]] += 1
                    self.non_zero_combination_prefix[bigram[0]].add(bigram[1])
                    self.non_zero_combination_suffix[bigram[1]].add(bigram[0])
        # set index for next word added to the clustering process
        self.next_word_index = max_vocab_size
        # calculate q, s and avg_mut_information of initial clusters
        self.q = np.zeros([max_vocab_size, max_vocab_size], dtype=float).tolist()
        for prefix_comb in self.non_zero_combination_prefix.keys():
            for suffix in self.non_zero_combination_prefix[prefix_comb]:
                tup = (prefix_comb, suffix)
                count_tuple = self.bigram_count[tup[0]][tup[1]]
                self.q[tup[0]][tup[1]] = (count_tuple/self.total_bigram_count) * math.log(
                    (count_tuple*self.total_bigram_count) /
                    (self.as_prefix_count[tup[0]]*self.as_suffix_count[tup[1]])
                )
        # calculate initial mutual information
        self.avg_mut_info = sum(map(sum, self.q))
        self.s = np.zeros(len(self.vocabulary), dtype=float).tolist()
        for cl in self.clusters:
            i = cl.cluster_id
            ql = 0
            qr = 0
            for cl_id in self.non_zero_combination_prefix[i]:
                ql += self.q[i][ cl_id]
            for cl_id in self.non_zero_combination_suffix[i]:
                qr += self.q[cl_id][ i]
            qs = self.q[i][i]
            self.s[cl.cluster_id] = ql+qr-qs
        # evaluate information loss for first merge
        inf_loss = np.ones([max_vocab_size, max_vocab_size], dtype=float)* 1000000
        self.avg_info_loss = inf_loss.tolist()
        for cl_a in self.clusters:
            for cl_b in self.clusters:
                if cl_a.cluster_id < cl_b.cluster_id:
                    self.avg_info_loss[cl_a.cluster_id][cl_b.cluster_id] =\
                        self.evaluate_merge(cl_a.cluster_id, cl_b.cluster_id)
        print("Starting merging process..")
        # get best pair for initial merge
        clusters_to_merge = self.get_min_avg_info_loss()
        # start merging process with best pair
        # keep splitting until corpus is reduced to desired number of clusters
        self.merge_clusters(clusters_to_merge[0], clusters_to_merge[1])
        # if optimization after greedy clustering is desired
        # for each word in the vocab try to find a different cluster, where avg_mut_info increases the most
        # if a word was moved during this process - repeat until no more words are moved
        self.save_clustering(out_file=self.out_file+'_pre_optimization')
        print("Initial clustering completed!")
        print(self.clusters)
        if optimization:
            print("Starting post optimization process..")
            self.post_cluster_optimization()
            print("Optimization completed!")
            self.save_clustering(out_file=self.out_file+'_final')

    def get_min_avg_info_loss(self):
        x = 0
        y = 0
        min = 100000
        for i in range(len(self.avg_info_loss)):
            for j in range(len(self.avg_info_loss[0])):
                if self.avg_info_loss[i][j]<min:
                    min = self.avg_info_loss[i][j]
                    x = i
                    y = j
        return x, y

    def save_clustering(self, out_file):
        """
        saves clustering into json file
        :param out_file: desired base_name
        :return:
        """
        base_path = path.abspath(path.dirname(__file__))
        base_path = base_path[:-17]
        base_path += 'clustering/'
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
        """
        adds next word in the queue into the clustering process by creating a new cluster and updating all counts
        :param cluster_id:
        :return:
        """
        # init cluster
        next_word = self.word_queue[self.next_word_index][0]
        self.clusters.append(Cluster(cluster_id,next_word))
        for cl in self.clusters:
            #iterate through all currently possible bigram combinations using introduced word
            for word in cl.words:
                cl_id  = self.get_cluster_id(word)
                count = self.word_bigram_count[(next_word,word)]
                #if bigram exist update corresponding counts
                if count > 0:
                    self.bigram_count[cluster_id][cl_id] += count
                    self.as_prefix_count[cluster_id] += count
                    self.as_suffix_count[cl_id] += count
                    self.non_zero_combination_prefix[cluster_id].add(cl_id)
                    self.non_zero_combination_suffix[cl_id].add(cluster_id)
                else:
                    self.word_bigram_count.pop((next_word, word))
                # prevent double add of cluster_id,cluster_id
                if cl_id != cluster_id:
                    count = self.word_bigram_count[(word,next_word)]
                    if count > 0:
                        self.bigram_count[cl_id][cluster_id] += count
                        self.as_prefix_count[cl_id] += count
                        self.as_suffix_count[cluster_id] += count
                        self.non_zero_combination_prefix[cl_id].add(cluster_id)
                        self.non_zero_combination_suffix[cluster_id].add(cl_id)
                    else:
                        self.word_bigram_count.pop((word,next_word))
        # calculate s and q values of new cluster
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
        # calculate info_loss of merges concerning new cluster
        for cl in self.clusters:
            if cluster_id != cl.cluster_id:
                if cl.cluster_id < cluster_id:
                    self.avg_info_loss[cl.cluster_id][cluster_id] =  self.evaluate_merge(cl.cluster_id,cluster_id)
                else:
                    self.avg_info_loss[cluster_id][cl.cluster_id] = self.evaluate_merge(cluster_id,cl.cluster_id)
        # increase index for word_queue
        self.next_word_index += 1

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
        """
        Calculate quality value of (cl_id_l,cl_id_r)
        :param cl_id_l:
        :param cl_id_r:
        :return: q[cl_id_l][cl_id_r]
        """
        cdef long p,pl,pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0.0

    def calc_q_l2(self, int cl_id_l, int cl_id_l2, int cl_id_r):
        """
        Calculate quality value of (cl_id_l+cl_id_l2,cl_id_r)
        :param cl_id_l:
        :param cl_id_l2:
        :param cl_id_r:
        :return:
        """
        cdef long p, pl, pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        p += self.bigram_count[cl_id_l2][ cl_id_r]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            pl += self.as_prefix_count[cl_id_l2]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0.0

    def calc_q_r2(self, int cl_id_l, int cl_id_r, int cl_id_r2):
        """
        Calculate quality value of (cl_id_l,cl_id_r+cl_id_r2)
        :param cl_id_l:
        :param cl_id_r:
        :param cl_id_r2:
        :return:
        """
        cdef long p,pl,pr
        p = self.bigram_count[cl_id_l][ cl_id_r]
        p += self.bigram_count[cl_id_l][ cl_id_r2]
        if p > 0:
            pl = self.as_prefix_count[cl_id_l]
            pr = self.as_suffix_count[cl_id_r]
            pr += self.as_suffix_count[cl_id_r2]
            return (p/self.total_bigram_count)*math.log((p*self.total_bigram_count)/(pl*pr))
        else:
            return 0.0

    def calc_q_full(self,int cl_id_l,int cl_id_l2,int cl_id_r,int cl_id_r2):
        """
        Calculate quality value of (cl_id_l+cl_id_l2,cl_id_r+cl_id_r2)
        :param cl_id_l:
        :param cl_id_l2:
        :param cl_id_r:
        :param cl_id_r2:
        :return:
        """
        cdef long p,pl,pr
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
            return 0.0

    def calc_q_temp(self, bigram_count, as_prefix_count, as_suffix_count, cl_id_l, cl_id_r):
        """
        Calculate quality value of (cl_id_l,cl_id_r) using temporary counts
        :param bigram_count: temporary bigram_count
        :param as_prefix_count: temporary prefix_count
        :param as_suffix_count: temporary suffix_count
        :param cl_id_l:
        :param cl_id_r:
        :return:
        """
        cdef long p,pl,pr
        p = bigram_count[cl_id_l][ cl_id_r]
        if p > 0:
            pl = as_prefix_count[cl_id_l]
            pr = as_suffix_count[cl_id_r]
            return (p / self.total_bigram_count) * math.log((p * self.total_bigram_count) / (pl * pr))
        return 0.0

    def merge_clusters(self, cluster_id_a, cluster_id_b):
        """
        Creates initial greedy clustering, starts by merging cluster_a with cluster_b to cluster_a'
        :param cluster_id_a:
        :param cluster_id_b:
        :return:
        """
        merge_counter = 1
        total_merges = len(self.vocabulary)-self.desired_num_clusters+1
        print("first merge pair:" + str((cluster_id_a, cluster_id_b))+ " merge: " +str(merge_counter)+ "/" + str(total_merges))
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
            # add next word to the clustering if there are words left in the queue
            if self.next_word_index < len(self.word_queue):
                self.add_next_word(cluster_id_b)

            # set clusters for next merge
            if len(self.clusters) > 1:
                best_merge_pair = self.get_min_avg_info_loss()
                merge_counter +=1
                print("next merge pair: " + str(best_merge_pair)+ " merge: " +str(merge_counter)+ "/" + str(total_merges))
                cluster_id_a = best_merge_pair[0]
                cluster_id_b = best_merge_pair[1]



    def cluster_reordering(self):
        """
        rearranges clusters and corresponding counts to get a dense representation after the initial clustering
        :return:
        """
        i = 0
        # remapping of the clusters
        mapping = dict()
        # create new count arrays
        new_q = np.zeros([len(self.clusters),len(self.clusters)],dtype=float).tolist()
        new_bigram_count = np.zeros([len(self.clusters),len(self.clusters)],dtype=int).tolist()
        # create mapping and rename clusters accordingly
        for cluster in self.clusters:
            mapping[cluster.cluster_id]=i
            cluster.set_cluster_id(i)
            i = i+1
        # iterate through all existing counts and move copy them to their new place in the new counts
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
        # rename the non_zero_combination information
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
        # use the new created counts
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
        # reorder clusters to dense arrangement to reduce copy overhead
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
        # initialize counts for a simulated cluster only containing 'word'
        tmp_bigram_count_prefix = np.zeros(len(self.clusters), dtype=int).tolist()
        tmp_bigram_count_suffix = np.zeros(len(self.clusters), dtype=int).tolist()
        tmp_bigram_count_self = 0
        tmp_as_prefix_count = 0
        tmp_as_suffix_count = 0
        tmp_non_zero_combination_prefix = set()
        tmp_non_zero_combination_suffix = set()
        # set the cluster_id of the cluster 'word' was previously contained in
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
        # create the cluster information for the simulated cluster
        for suffix in self.word_non_zero_combination_prefix[word]:
            suffix_id = self.get_cluster_id(suffix)
            obsolete_bigram = self.word_bigram_count[(word, suffix)]
            if suffix != word:
                tmp_bigram_count_prefix[suffix_id] += obsolete_bigram
                tmp_as_prefix_count += obsolete_bigram
                tmp_non_zero_combination_prefix.add(suffix_id)
            else:
                tmp_bigram_count_self += obsolete_bigram
                tmp_as_prefix_count += obsolete_bigram
                tmp_as_suffix_count += obsolete_bigram
                tmp_non_zero_combination_prefix.add(-1)
                tmp_non_zero_combination_suffix.add(-1)
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
                tmp_bigram_count_suffix[prefix_id] += obsolete_bigram
                tmp_as_suffix_count += obsolete_bigram
                tmp_non_zero_combination_suffix.add(prefix_id)
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
                                                                 post_del_info=post_del_info, tmp_bigram_count_prefix=tmp_bigram_count_prefix,tmp_bigram_count_suffix=tmp_bigram_count_suffix,tmp_bigram_count_self=tmp_bigram_count_self,tmp_as_prefix_count=tmp_as_prefix_count,tmp_as_suffix_count=tmp_as_suffix_count,tmp_non_zero_combination_prefix=tmp_non_zero_combination_prefix,tmp_non_zero_combination_suffix=tmp_non_zero_combination_suffix) - self.avg_mut_info
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


    def evaluate_move(self, word, cluster_id, post_del_info, tmp_bigram_count_prefix, tmp_bigram_count_suffix, tmp_bigram_count_self, tmp_as_suffix_count, tmp_as_prefix_count, tmp_non_zero_combination_prefix, tmp_non_zero_combination_suffix):
        """
        calculates the avg_mut_info after moving word into new cluster
        :param word: word that is evaluated
        :param cluster_id: cluster it could be moved to
        :param post_del_info: mutual information without this word
        :param tmp_bigram_count_prefix: bigram information of word (word,*)
        :param tmp_bigram_count_suffix: bigram information of word (*,word)
        :param tmp_bigram_count_self: bigram information of (word,word)
        :param tmp_as_suffix_count: suffix count of word
        :param tmp_as_prefix_count: prefix count of word
        :param tmp_non_zero_combination_prefix: bigram combinations with word as prefix
        :param tmp_non_zero_combination_suffix: bigram combinations with word as suffix
        :return:
        """
        # create copies of the non zero combinations of the considered cluster that are later merged with combinations of the moved word
        non_zero_combination_prefix = self.non_zero_combination_prefix[cluster_id].copy()
        non_zero_combination_suffix = self.non_zero_combination_suffix[cluster_id].copy()

        # remove q values using the cluster from avg_mut_info we want to move to, these will later be recalculated
        for cl_id in non_zero_combination_prefix:
            post_del_info -= self.q[cluster_id][ cl_id]
        for cl_id in non_zero_combination_suffix:
            if cl_id != cluster_id:
                post_del_info -= self.q[cl_id][ cluster_id]
        # update the non_zero counts -1 denotes the simulated cluster of 'word'
        for cl_id in tmp_non_zero_combination_prefix:
            if cl_id != -1:
                non_zero_combination_prefix.add(cl_id)
            else:
                non_zero_combination_prefix.add(cluster_id)
        for cl_id in tmp_non_zero_combination_suffix:
            if cl_id != -1:
                non_zero_combination_suffix.add(cl_id)
            else:
                non_zero_combination_suffix.add(cluster_id)
        ###
        # To future self or anyone maintaining this code: This is needed because after removing a word in move_to_best_cluster we don't save the information, that a cluster was involved in bigrams of form (cluster_id,removed_word). this information is partially recovered by the deleted word, when combining the non_zero_combinations
        # however this combination will only be found in non_zero_combination_suffix(prefix if bigram was of form (removed_word,cluster_id)). As we only calculate q values of (cluster_id,cluster_id) using the prefix combos to prevent double addition we need to make sure that the info is also present in the prefix combos
        if cluster_id in non_zero_combination_prefix:
            non_zero_combination_suffix.add(cluster_id)
        if cluster_id in non_zero_combination_suffix:
            non_zero_combination_prefix.add(cluster_id)
        ###
        # add updated q values
        cdef int p,pl,pr
        for cl_id in non_zero_combination_prefix:
            if cl_id != cluster_id:
                p = self.bigram_count[cluster_id][cl_id]+tmp_bigram_count_prefix[cl_id]
                pl = self.as_prefix_count[cluster_id]+tmp_as_prefix_count
                pr = self.as_suffix_count[cl_id]
                post_del_info += (p / self.total_bigram_count) * math.log((p * self.total_bigram_count) / (pl * pr))
            else:
                p = self.bigram_count[cluster_id][cluster_id]+tmp_bigram_count_self + tmp_bigram_count_prefix[cluster_id]+tmp_bigram_count_suffix[cluster_id]
                pl = self.as_prefix_count[cluster_id]+tmp_as_prefix_count
                pr = self.as_suffix_count[cluster_id]+tmp_as_suffix_count
                post_del_info += (p / self.total_bigram_count) * math.log((p * self.total_bigram_count) / (pl * pr))
        for cl_id in non_zero_combination_suffix:
            if cl_id != cluster_id:
                p = self.bigram_count[cl_id][cluster_id]+tmp_bigram_count_suffix[cl_id]
                pl = self.as_prefix_count[cl_id]
                pr = self.as_suffix_count[cluster_id]+tmp_as_suffix_count
                post_del_info += (p / self.total_bigram_count) * math.log((p * self.total_bigram_count) / (pl * pr))
        return post_del_info

    def perform_move(self, word, cluster_id, post_del_info):
        """
        Moves a word to a new cluster, similar as evaluate_move put retains performed changes
        :param word:
        :param cluster_id:
        :param post_del_info:
        :return:
        """
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


