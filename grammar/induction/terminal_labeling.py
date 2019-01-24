from abc import ABCMeta, abstractmethod
from collections import defaultdict
from hybridtree.monadic_tokens import MonadicToken
from discodop.lexicon import getunknownwordmodel, unknownword4, replaceraretestwords, YEARRE, NUMBERRE, UNK


class TerminalLabeling:
    __metaclass__ = ABCMeta

    @abstractmethod
    def token_label(self, token, _loc=None):
        """
        :type token: MonadicToken
        :type _loc: int
        """
        pass

    def token_tree_label(self, token):
        if token.type() == "CONLL-X":
            return self.token_label(token) + " : " + token.deprel()
        elif token.type() == "CONSTITUENT-CATEGORY":
            return token.category() + " : " + token.edge()
        else:
            return self.token_label(token) + " : " + token.edge()

    def prepare_parser_input(self, tokens):
        return [self.token_label(token, _loc) for _loc, token in enumerate(tokens)]

    def serialize(self):
        return {'type': self.__class__.__name__}

    @staticmethod
    @abstractmethod
    def deserialize(json_object):
        pass

# class ConstituentTerminalLabeling(TerminalLabeling):
#     def token_label(self, token):
#         if isinstance(token, ConstituentTerminal):
#             return token.pos()
#         elif isinstance(token, ConstituentCategory):
#             return token.category()
#         else:
#             assert False


class FeatureTerminals(TerminalLabeling):
    def __init__(self, token_to_features, feature_filter):
        self.token_to_features = token_to_features
        self.feature_filter = feature_filter

    def token_label(self, token, _loc=None):
        isleaf = token.type() == "CONSTITUENT-TERMINAL"
        feat_list = self.token_to_features(token, isleaf)
        features = self.feature_filter([feat_list])
        return "[" + (",".join([str(key) + ':' + str(val) for key, val in features]))\
            .translate(str.maketrans('', '', ' ')) + "]"

    def __str__(self):
        return "feature-terminals"


class FrequencyBiasedTerminalLabeling(TerminalLabeling):
    def __init__(self, fine_labeling, fall_back, corpus=None, threshold=4, fine_label_count=None):
        self.fine_labeling = fine_labeling
        self.fall_back = fall_back
        self.backoff_mode = False
        self.threshold = threshold

        if fine_label_count is None:
            self.fine_label_count = defaultdict(lambda: 0)
            for tree in corpus:
                for token in tree.token_yield():
                    label = self.fine_labeling.token_label(token)
                    self.fine_label_count[label] += 1
            self.fine_label_count = frozenset({label for label in self.fine_label_count if self.fine_label_count[label] >= threshold})
        else:
            self.fine_label_count = fine_label_count

    def token_label(self, token, _loc=None):
        fine_label = self.fine_labeling.token_label(token)
        if not self.backoff_mode and fine_label in self.fine_label_count:
            return fine_label
        else:
            return self.fall_back.token_label(token)

    def __str__(self):
        return "frequency-biased["\
               + str(self.threshold) \
               + '|' + str(self.fine_labeling) \
               + '|' + str(self.fall_back) + "]"

    def serialize(self):
        return {'type': self.__class__.__name__,
                'threshold': self.threshold,
                'fine lexicon': [x for x in self.fine_label_count],
                'fine labeling': self.fine_labeling.serialize(),
                'fallback labeling': self.fall_back.serialize()}

    @staticmethod
    def deserialize(json_object):
        fine = deserialize_labeling(json_object['fine labeling'])
        fall_back = deserialize_labeling(json_object['fallback labeling'])
        fine_lexicon = frozenset({x for x in json_object['fine lexicon']})
        return FrequencyBiasedTerminalLabeling(
            fine_labeling=fine,
            fall_back=fall_back,
            fine_label_count=fine_lexicon
        )


class CompositionalTerminalLabeling(TerminalLabeling):
    def __init__(self, first_labeling, second_labeling, binding_string="__+__"):
        self.first_labeling = first_labeling
        self.second_labeling = second_labeling
        self.binding_string = binding_string

    def __str__(self):
        return str(self.first_labeling) + '-' + str(self.second_labeling)

    def token_label(self, token, _loc=None):
        first = self.first_labeling.token_label(token, _loc)
        second = self.second_labeling.token_label(token, _loc)
        return first + self.binding_string + second

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'first': self.first_labeling.serialize(),
            'second': self.second_labeling.serialize(),
            'binding string': self.binding_string
        }

    @staticmethod
    def deserialize(json_object):
        first = deserialize_labeling(json_object['first'])
        second = deserialize_labeling(json_object['second'])
        return CompositionalTerminalLabeling(first, second, json_object['binding string'])


class FormTerminals(TerminalLabeling):
    def token_label(self, token, _loc=None):
        return token.form()

    def __str__(self):
        return 'form'

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'FormTerminals'
        return FormTerminals()


class CPosTerminals(TerminalLabeling):
    def token_label(self, token, _loc=None):
        return token.cpos()

    def __str__(self):
        return 'cpos'

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'CPosTerminals'
        return CPosTerminals()


class PosTerminals(TerminalLabeling):
    def token_label(self, token, _loc=None):
        return token.pos()

    def __str__(self):
        return 'pos'

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'PosTerminals'
        return PosTerminals()


class CPOS_KON_APPR(TerminalLabeling):
    def token_label(self, token, _loc=None):
        cpos = token.pos()
        if cpos in ['KON', 'APPR']:
            return cpos + token.form().lower()
        else:
            return cpos

    def __str__(self):
        return 'cpos-KON-APPR'

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'CPOS_KON_APPR'
        return CPOS_KON_APPR()


class FormTerminalsUnk(TerminalLabeling):
    def __init__(self, trees, threshold, UNK="UNKNOWN", filter=[]):
        """
        :param trees: corpus of trees
        :param threshold: UNK words below the threshold
        :type threshold: int
        :param UNK: representation string of UNK in grammar
        :type UNK: str
        :param filter: a list of POS tags which are always UNKed
        :type filter: list[str]
        """
        self.__terminal_counts = {}
        self.__UNK = UNK
        self.__threshold = threshold
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    key = token.form().lower()
                    if key in self.__terminal_counts:
                        self.__terminal_counts[key] += 1
                    else:
                        self.__terminal_counts[key] = 1

    def __str__(self):
        return 'form-unk-' + str(self.__threshold)

    def token_label(self, token, _loc=None):
        form = token.form().lower()
        if self.__terminal_counts.get(form, 0) < self.__threshold:
            form = self.__UNK
        return form


class FormTerminalsPOS(TerminalLabeling):
    def __init__(self, trees, threshold, filter=[]):
        """
        :param trees: corpus of trees
        :param threshold: UNK words below the threshold
        :type threshold: int
        :param UNK: representation string of UNK in grammar
        :type UNK: str
        :param filter: a list of POS tags which are always UNKed
        :type filter: list[str]
        """
        self.__terminal_counts = {}
        self.__threshold = threshold
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    key = token.form().lower()
                    if key in self.__terminal_counts:
                        self.__terminal_counts[key] += 1
                    else:
                        self.__terminal_counts[key] = 1

    def __str__(self):
        return 'form-POS-' + str(self.__threshold)

    def token_label(self, token, _loc=None):
        form = token.form().lower()
        if self.__terminal_counts.get(form, 0) < self.__threshold:
            form = token.pos()
        return form


class FormPosTerminalsUnk(TerminalLabeling):
    def __init__(self, trees, threshold, UNK="UNKNOWN", filter=[]):
        """
        :param trees: corpus of trees
        :param threshold: UNK words below the threshold
        :type threshold: int
        :param UNK: representation string of UNK in grammar
        :type UNK: str
        :param filter: a list of POS tags which are always UNKed
        :type filter: list[str]
        """
        self.__terminal_counts = {}
        self.__UNK = UNK
        self.__threshold = threshold
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    key = (token.form().lower(), token.pos())
                    if key in self.__terminal_counts:
                        self.__terminal_counts[key] += 1
                    else:
                        self.__terminal_counts[key] = 1

    def __str__(self):
        return 'form-pos-unk-' + str(self.__threshold) + '-pos'

    def token_label(self, token, _loc=None):
        pos = token.pos()
        form = token.form().lower()
        if self.__terminal_counts.get((form, pos), 0) < self.__threshold:
            form = self.__UNK
        return form + '-:-' + pos


class FormPosTerminalsUnkMorph(TerminalLabeling):
    def __init__(self, trees, threshold, UNK="UNKNOWN", filter=[], add_morph={}):
        self.__terminal_counts = defaultdict(lambda: 0)
        self.__UNK = UNK
        self.__threshold = threshold
        self.__add_morph = add_morph
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    self.__terminal_counts[(token.form().lower(), token.pos())] += 1

    def __str__(self):
        return 'form-pos-unk-' + str(self.__threshold) + '-morph-pos'

    def token_label(self, token, _loc=None):
        pos = token.pos()
        form = token.form().lower()
        if self.__terminal_counts.get((form, pos)) < self.__threshold:
            form = self.__UNK
            if pos in self.__add_morph:
                feats = map(lambda x: tuple(x.split('=')), token.feats().split('|'))
                for feat in feats:
                    if feat[0] in self.__add_morph[pos]:
                        form += '#' + feat[0] + ':' + feat[1]
        return form + '-:-' + pos


class Suffix(TerminalLabeling):
    def __init__(self, threshold, trees=None, counts=None, UNK="UNKNOWN", suffix_length=2):
        self.__trees = trees
        self.__terminal_counts = defaultdict(lambda: 0)
        self.__UNK = UNK
        self.__threshold = threshold
        self.__suffix_len = suffix_length
        if trees is not None:
            for tree in trees:
                for token in tree.token_yield():
                    self.__terminal_counts[token.form().lower() + "-" + token.pos()] += 1
        else:
            self.__terminal_counts = counts


    def __str__(self):
        return "suffix-" + str(self.__threshold)

    def token_label(self, token, _loc=None):
        form = token.form().lower()
        pos = token.pos()
        pre_label = form +"-"+ pos
        if self.__terminal_counts.get(pre_label) is not None:
            if self.__terminal_counts.get(pre_label) >= self.__threshold:
                return pre_label
        if pos in ['CARD', 'FM', 'XY']:
            return pos
        else:
            return form[-self.__suffix_len:].lower()


    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'unknown_threshold': self.__threshold,
            'unknown_base_label': self.__UNK,
            'terminal_counts': self.__terminal_counts,
            'suffix_len': self.__suffix_len
        }

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'Suffix'
        return Suffix(counts=json_object['terminal_counts'],
                      threshold=json_object['unknown_threshold'],
                      UNK=json_object['unknown_base_label'],
                      suffix_length=json_object['suffix_len'])




class UNK4(TerminalLabeling):
    def __init__(self, threshold, trees=None, UNK="UNKNOWN", use_pos=False, terminal_counts=None):
        self.__trees = trees
        self.__terminal_counts = defaultdict(lambda: 0)
        self.__UNK = UNK
        self.__threshold = threshold
        self.__use_pos = use_pos
        if trees is not None:
            if self.__use_pos:
                for tree in trees:
                    for token in tree.token_yield():
                        self.__terminal_counts[(token.form().lower()+"-"+token.pos())] += 1
            else:
                for tree in trees:
                    for token in tree.token_yield():
                        self.__terminal_counts[token.form().lower()] +=1
        else:
            self.__terminal_counts = terminal_counts

    def __str__(self):
        return "unknownword4-" + str(self.__threshold)

    def token_label(self, token, _loc=None):
        form = token.form().lower()
        pos = token.pos()
        if self.__use_pos:
            if self.__terminal_counts.get(form +"-"+ pos) is not None:
                if self.__terminal_counts.get(form +"-"+pos) >= self.__threshold:
                    return form + "-" + pos
        else:
            if self.__terminal_counts.get(form) is not None:
                if self.__terminal_counts.get(form) >= self.__threshold:
                    return form

        word = token.form()
        sig = self.__UNK
        has_lower = False
        has_letter = False
        has_digit = False
        if word[0].isupper():  #check capitalization
            all_caps = True
            for letter in word:
                if letter.islower():
                    all_caps = False
                    break;
            if all_caps:
                sig += "-AC"
            elif _loc == 0:
                sig += "-SC" # loc?
            else:
                sig += "-C"
        for letter in word:
            if letter.isdigit():
                has_digit = True
                break
        for letter in word:
            if letter.islower():
                has_lower = True
                break
            elif letter.isupper():
                has_letter = True
        if has_digit:
            sig += "-n"
        else:
            sig += "-N"
        if has_lower:
            sig += "-L"
        elif has_letter:
            sig += "-U"
        else:
            sig += "-S"
        if "-" in word:
            sig += "-H"
        if "." in word:
            sig += "-P"
        if "," in word:
            sig += "-C"
        if len(word) > 3:
            if word[-1].islower() or word[-1].isupper():
                sig += "-%s" % word[-2:].lower()
        return sig

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'unknown_threshold': self.__threshold,
            'unknown_base_label': self.__UNK,
            'terminal_counts': self.__terminal_counts,
            'use_pos': self.__use_pos
        }

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'UNK4'
        return UNK4(terminal_counts=json_object['terminal_counts'],
                    threshold=json_object['unknown_threshold'],
                    UNK=json_object['unknown_base_label'],
                    use_pos=json_object['use_pos'])



class StanfordUNKing(TerminalLabeling):
    def __init__(self, trees=None, unknown_threshold=4, openclass_threshold=150, data=None):
        self.unknown_threshold = unknown_threshold
        self.openclass_threshold = openclass_threshold
        self.backoff_mode = False
        if data:
            self.openclasswords, self.sigs, self.words, self.lexicon = data
        else:
            sentences = []
            for tree in trees:
                sentence = []
                for token in tree.token_yield():
                    sentence.append((token.form(), token.pos()))
                sentences.append(sentence)
            (sigs, words, lexicon, wordsfortag, openclasstags,
                openclasswords, tags, wordtags,
                wordsig, sigtag), msg \
                = getunknownwordmodel(sentences, unknownword4, self.unknown_threshold, self.openclass_threshold)
            self.openclasswords = openclasswords
            self.sigs = sigs
            self.words = words
            self.lexicon = lexicon

    def __str__(self):
        return "stanford-unk-" + str(self.unknown_threshold) \
               + "-openclass-" + str(self.openclass_threshold)

    def token_label(self, token, _loc=None):
        word = token.form()

        # adapted from discodop
        if YEARRE.match(word):
            return '1970'
        elif NUMBERRE.match(word):
            return '000'
        if not self.backoff_mode:
            if word in self.lexicon:
                return word
            elif word.lower() in self.lexicon:
                return word.lower()
            else:
                sig = unknownword4(word, _loc, self.lexicon)
                if sig in self.sigs:
                    return sig
                else:
                    return UNK
        else:
            if word in self.lexicon and word not in self.openclasswords:
                return word
            elif word.lower() in self.lexicon and word not in self.openclasswords:
                return word.lower()
            else:
                sig = unknownword4(word, _loc, self.lexicon)
                if sig in self.sigs:
                    return sig
                else:
                    return UNK

    def serialize(self):
        return {
            'type': self.__class__.__name__,
            'unknown_threshold': self.unknown_threshold,
            'openclass_threshold': self.openclass_threshold,
            'openclasswords': self.openclasswords,
            'sigs': self.sigs,
            'words': self.words,
            'lexicon': self.lexicon
        }

    @staticmethod
    def deserialize(json_object):
        assert json_object['type'] == 'StanfordUNKing'
        openclasswords = json_object['openclasswords']
        sigs = json_object['sigs']
        words = json_object['words']
        lexicon = json_object['lexicon']
        return StanfordUNKing(unknown_threshold=json_object['unknown_threshold'],
                              openclass_threshold=json_object['openclass_threshold'],
                              data=(openclasswords, sigs, words, lexicon))


class TerminalLabelingFactory:
    def __init__(self):
        self.__strategies = {}

    def register_strategy(self, name, strategy):
        """
        :type name: str
        :type strategy: TerminalLabeling
        """
        self.__strategies[name] = strategy

    def get_strategy(self, name):
        """
        :type name: str
        :rtype: TerminalLabeling
        """
        return self.__strategies[name]


def the_terminal_labeling_factory():
    """
    :rtype : TerminalLabelingFactory
    """
    factory = TerminalLabelingFactory()
    factory.register_strategy('form', FormTerminals())
    factory.register_strategy('pos', PosTerminals())
    factory.register_strategy('cpos', CPosTerminals())
    factory.register_strategy('cpos-KON-APPR', CPOS_KON_APPR())
    return factory


def deserialize_labeling(json_object):
    return globals().get(json_object['type']).deserialize(json_object)
