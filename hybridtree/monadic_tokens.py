__author__ = 'kilian'

from abc import ABCMeta, abstractmethod


class MonadicToken:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rank(self):
        """
        :rtype: int
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        :rtype: str
        """
        pass


class CoNLLToken(MonadicToken):
    def __init__(self, form, lemma, cpos, pos, feats, deprel):
        super(CoNLLToken, self).__init__()
        self.__form = form
        self.__lemma = lemma
        self.__cpos = cpos
        self.__pos = pos
        self.__feats = feats
        self.__deprel = deprel

    def rank(self):
        return 1

    def form(self):
        return self.__form

    def lemma(self):
        return self.__lemma

    def cpos(self):
        return self.__cpos

    def pos(self):
        return self.__pos

    def feats(self):
        return self.__feats

    def deprel(self):
        return self.__deprel

    def set_deprel(self, deprel):
        self.__deprel = deprel

    def __str__(self):
        return self.form() + ' : ' + self.pos() + ' : ' + self.deprel()

    def __eq__(self, other):
        if not isinstance(other, CoNLLToken):
            return False
        else:
            return all([self.form() == other.form()
                           , self.cpos() == other.cpos()
                           , self.pos() == other.pos()
                           , self.feats() == other.feats()
                           , self.lemma() == other.lemma()
                           , self.deprel() == other.deprel()])
4

class ConstituencyToken(MonadicToken):
    def __init__(self):
        super(ConstituencyToken, self).__init__()

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class ConstituentTerminal(ConstituencyToken):
    def __init__(self, form, pos):
        super(ConstituentTerminal, self).__init__()
        self.__form = form
        self.__pos = pos

    def rank(self):
        return 0

    def form(self):
        return self.__form

    def pos(self):
        return self.__pos

    def __str__(self):
        return self.form() + ' : ' + self.pos()


class ConstituentCategory(ConstituencyToken):
    def __init__(self, category):
        super(ConstituentCategory, self).__init__()
        self.__category = category

    def rank(self):
        return 1

    def category(self):
        return self.__category

    def __str__(self):
        return self.category()


def construct_conll_token(form, pos, _=True):
    return CoNLLToken(form, '_', pos, pos, '_', '_')


def construct_constituent_token(form, pos, terminal):
    if terminal:
        return ConstituentTerminal(form, pos)
    else:
        return ConstituentCategory(form)