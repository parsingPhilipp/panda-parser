__author__ = 'kilian'

from abc import ABCMeta, abstractmethod


class BiRankedToken:
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


class CoNLLToken(BiRankedToken):
    def __init__(self, form, lemma, pos, deprel):
        super(CoNLLToken, self).__init__()
        self.__form = form
        self.__lemma = lemma
        self.__pos = pos
        self.__deprel = deprel

    def rank(self):
        return 1

    def form(self):
        return self.__form

    def lemma(self):
        return self.__lemma

    def pos(self):
        return self.__pos

    def deprel(self):
        return self.__deprel

    def set_deprel(self, deprel):
        self.__deprel = deprel

    def __str__(self):
        return self.form() + ' : ' + self.pos() + ' : ' + self.deprel()

    def __eq__(self, other):
        return all([self.form() == other.form()
                    , self.pos() == other.pos()
                    , self.lemma() == other.lemma()
                    , self.deprel() == other.deprel()
                    ])


class ConstituencyToken(BiRankedToken):
    def __init__(self):
        super(ConstituencyToken, self).__init__()

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class ConstituencyTerminal(ConstituencyToken):
    def __init__(self, form, pos):
        super(ConstituencyTerminal, self).__init__()
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


class ConstituencyCategory(ConstituencyToken):
    def __init__(self, category):
        super(ConstituencyCategory, self).__init__()
        self.__category = category

    def rank(self):
        return 1

    def category(self):
        return self.__category

    def __str__(self):
        return self.category()


def construct_dependency_token(form, pos, _):
    return CoNLLToken(form, '_', pos, '_')


def construct_constituent_token(form, pos, terminal):
    if terminal:
        return ConstituencyTerminal(form, pos)
    else:
        return ConstituencyCategory(form)