# Auxiliary classes to determine parsing accuracy.


# Keep records needed to compute recall/precision.
class ParseAccuracy(object):
    # Constructor.
    def __init__(self):
        # Sum of sentence precisions
        self.__precisions = 0
        # Sum of sentence recalls
        self.__recalls = 0
        # Sum of sentence F-measures
        self.__fmeasures = 0
        # Number of sentences
        self.__n = 0
        # Number of sentences where parsing failed.
        self.__n_failures = 0

        self.__true_positives = 0
        self.__false_negetives = 0
        self.__false_positives = 0

        self.__exact_match = 0

    # Compare two sets of lists. Process accuracy.
    # retrieved: set of comparable objects
    # correct: set of comparable objects
    def add_accuracy(self, retrieved, relevant):
        inters = retrieved & relevant

        # happens in case of parse failure
        # there are two options here:
        #   - parse failure -> no spans at all, thus precision = 1
        #   - parse failure -> a dummy tree with all spans wrong, thus precision = 0
        precision = 1.0 * len(inters) / len(retrieved) \
            if len(retrieved) > 0 else 0
        recall = 1.0 * len(inters) / len(relevant) \
            if len(relevant) > 0 else 0
        fmeasure = 2.0 * precision * recall / (precision + recall) \
            if precision + recall > 0 else 0

        self.__true_positives += len(inters)
        self.__false_negetives += len(relevant) - len(inters)
        self.__false_positives += len(retrieved) - len(inters)

        self.__precisions += precision
        self.__recalls += recall
        self.__fmeasures += fmeasure
        self.__n += 1

        if len(inters) == len(retrieved) and len(inters) == len(relevant):
            self.__exact_match += 1

    # Count one more failure.
    def add_failure(self, correct=set()):
        self.__n_failures += 1

    # Get number of instances.
    # return: int
    def n(self):
        return self.__n

    # Number of sentences where parsing failed.
    # return: int
    def n_failures(self):
        return self.__n_failures

    # Average precision.
    # return: float
    def avrg_precision(self):
        return self.__precisions / self.__n

    def precision(self):
        return self.__true_positives * 1.0 / (self.__true_positives + self.__false_positives)

    # Average recall.
    # return: float
    def avrg_recall(self):
        return self.__recalls / self.__n

    def recall(self):
        return self.__true_positives * 1.0 / (self.__true_positives + self.__false_negetives)

    # Average F-measure.
    # return: float
    def avrg_fmeasure(self):
        return self.__fmeasures / self.__n

    def fmeasure(self):
        return 2.0 * self.precision() * self.recall() / (self.precision() + self.recall())

    def exact_match(self):
        return 1.0 * self.__exact_match / self.__n


# Keep records needed to compute recall/precision.
class ParseAccuracyPenalizeFailures(ParseAccuracy):
    def __init__(self):
        super(ParseAccuracyPenalizeFailures, self).__init__()

    # Count one more failure.
    def add_failure(self, correct=set()):
        super(ParseAccuracyPenalizeFailures, self).add_failure()
        self.add_accuracy(set(), correct)


class PredicateArgumentScoring:
    def __init__(self):
        self.__labeled_dependency_scorer = ParseAccuracyPenalizeFailures()
        self.__unlabeled_dependency_scorer = ParseAccuracyPenalizeFailures()
        self.__labeled_frame_scorer = ParseAccuracyPenalizeFailures()
        self.__unlabeled_frame_scorer = ParseAccuracyPenalizeFailures()

    def add_failure(self, correct_frames=set()):
        self.__labeled_dependency_scorer.add_failure(
            self.extract_dependencies_from_frames(correct_frames, include_label=True)
        )
        self.__unlabeled_dependency_scorer.add_failure(
            self.extract_dependencies_from_frames(correct_frames, include_label=False)
        )
        self.__labeled_frame_scorer.add_failure(correct_frames)
        self.__unlabeled_frame_scorer.add_failure(self.extract_unlabeled_frames(correct_frames))

    def add_accuracy_frames(self, found, correct):
        self.__labeled_dependency_scorer.add_accuracy(
            self.extract_dependencies_from_frames(found, include_label=True)
            , self.extract_dependencies_from_frames(correct, include_label=True))
        self.__unlabeled_dependency_scorer.add_accuracy(
            self.extract_dependencies_from_frames(found, include_label=False)
            , self.extract_dependencies_from_frames(correct, include_label=False))
        self.__labeled_frame_scorer.add_accuracy(found, correct)
        self.__unlabeled_frame_scorer.add_accuracy(
            self.extract_unlabeled_frames(found)
            , self.extract_unlabeled_frames(correct))

    @staticmethod
    def extract_dependencies_from_frames(frames, include_label):
        dependencies = set()
        for predicate, arg_role_pairs in frames:
            for argument, role in arg_role_pairs:
                if include_label:
                    dependencies.add((predicate, argument, role))
                else:
                    dependencies.add((predicate, argument))
        return dependencies

    @staticmethod
    def extract_unlabeled_frames(frames):
        return {(predicate, frozenset({argument for argument, _ in arg_role_pairs}))
                for predicate, arg_role_pairs in frames}

    @property
    def labeled_dependency_scorer(self):
        return self.__labeled_dependency_scorer

    @property
    def unlabeled_dependency_scorer(self):
        return self.__unlabeled_dependency_scorer

    @property
    def labeled_frame_scorer(self):
        return self.__labeled_frame_scorer

    @property
    def unlabeled_frame_scorer(self):
        return self.__unlabeled_frame_scorer


__all__ = ["ParseAccuracy", "ParseAccuracyPenalizeFailures", "PredicateArgumentScoring"]
