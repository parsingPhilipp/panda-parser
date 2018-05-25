__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from grammar.lcfrs import LCFRS_rule


class AbstractDerivation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def root_id(self):
        """
        :rtype: str
        """
        pass

    @abstractmethod
    def getRule(self, id):
        """
        :param id:
        :type id:
        :return:
        :rtype: LCFRS_rule
        """
        pass

    @abstractmethod
    def child_ids(self, id):
        """
        :param id:
        :rtype: list[object]
        """
        pass

    @abstractmethod
    def child_id(self, id, i):
        pass

    @abstractmethod
    def position_relative_to_parent(self, id):
        """
        :param id:
        :type id: object
        :rtype: (object, int)
        """
        pass

    @abstractmethod
    def ids(self):
        """
        :rtype: list[object]
        """
        pass

    def __eq__(self, other):
        if not isinstance(other, AbstractDerivation):
            return False
        return self.__compare_recursive(self.root_id(), other, other.root_id())

    def __compare_recursive(self, id_self, other, id_other):
        if self.getRule(id_self) != other.getRule(id_other):
            return False
        if len(self.child_ids(id_self)) != len(other.child_ids(id_other)):
            return False
        for child_self, child_other in zip(self.child_ids(id_self), other.child_ids(id_other)):
            if not self.__compare_recursive(child_self, other, child_other):
                return False
        return True