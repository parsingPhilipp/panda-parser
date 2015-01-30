__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from lcfrs import *

class AbstractParser:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, grammar, input):
        """
        :type grammar: LCFRS
        :type input: list[str]
        :return:
        """
        pass

    @abstractmethod
    def best(self):
        """
        :rtype: float
        """
        pass

    @abstractmethod
    def recognized(self):
        """
        :rtype: bool
        """
        pass

    @abstractmethod
    def best_derivation_tree(self):
        """
        :rtype: AbstractDerivation
        :return:
        """
        pass
