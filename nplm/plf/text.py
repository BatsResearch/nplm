import re

import numpy as np


class WeakRule:
    def __init__(self, exec_module=None, label_maps=None,
                 name=None):
        # Label Map is
        # mapping from output of execution module to label groups
        # {0:[1,2,3],1:[3,4]}
        self.name = name
        self.label_maps = label_maps
        # Execution module is a function operates on single instances
        self.exec_module = exec_module

        self.true_votes = None

    def set_exec_module(self, exec_module):
        self.exec_module = exec_module

    def set_label_maps(self, label_maps):
        self.label_maps = label_maps

    def set_name(self, name):
        self.name = name

    def execute(self, data):
        if self.exec_module is None:
            raise NotImplementedError('Exec Module Not Specified')
        results = np.zeros(len(data))
        for idx, inst in enumerate(data):
            results[idx] = self.exec_module(inst)
        self.curr_results = results
        return results

    def eval(self, true_labels,
             class_wise_acc=False):

        if self.curr_results is None:
            raise ValueError("No Results Found!")
        if len(true_labels) != len(self.curr_results):
            raise ValueError("Incomparable Shape for GT")
        concerned_index = np.where(self.curr_results >= 0)[0]
        translated_votes = [[] for _ in range(len(self.curr_results))]
        for inst_idx, label in enumerate(true_labels):
            for vote, group in self.label_maps.items():
                if label in group:
                    translated_votes[inst_idx].append(vote)
        truncated_translated_votes = [item for idx, item in enumerate(translated_votes) if self.curr_results[idx] != -1]
        curr_votes = self.curr_results[concerned_index]
        self.true_votes = truncated_translated_votes
        polars = np.unique(curr_votes)
        precision = []
        for g in polars:
            idx = np.where(curr_votes == g)[0]
            cw_acc = 0
            for item_idx, vote in enumerate(curr_votes[idx]):
                if vote in truncated_translated_votes[idx[item_idx]]:
                    cw_acc += 1
            precision.append(cw_acc / len(idx))
        overall_precision = np.mean(precision)

        cw_acc = 0
        for item_idx, vote in enumerate(self.curr_results):
            if vote in translated_votes[item_idx]:
                cw_acc += 1
        overall_recall = cw_acc / len(self.curr_results)

        acc = 0
        for item_idx, vote in enumerate(curr_votes):
            if vote in truncated_translated_votes[item_idx]:
                acc += 1
        try:
            acc /= len(curr_votes)
        except ZeroDivisionError:
            acc = 0

        if class_wise_acc:
            class_wise_acc = []
            unique_class = np.unique(true_labels)
            sorted(unique_class)
            for label in unique_class:
                class_acc = 0
                idx = np.where(np.array(true_labels)[concerned_index] == label)[0]
                if len(idx) > 0:
                    for item_idx, vote in enumerate(curr_votes[idx]):
                        if vote in truncated_translated_votes[idx[item_idx]]:
                            class_acc += 1
                    class_wise_acc.append(class_acc / len(idx))
                else:
                    class_wise_acc.append(float('nan'))
            return acc, overall_precision, overall_recall, class_wise_acc

        return acc, overall_precision, overall_recall


class SpacyDepMatcherRule(WeakRule):
    def __init__(self, patterns, matcher, preproc=None, label_maps=None):
        if type(patterns) is not list and type(patterns) is dict:
            patterns = [patterns]

        def matcher(inst):
            if preproc is not None:
                inst = preproc(inst)
            matcher.add("", patterns)
            match = matcher(inst)
            res = False
            for r in match:
                res = res and len(r[1]) > 0
            return int(res)

        super().__init__(exec_module=matcher, label_maps=label_maps)


class BinaryRERules(WeakRule):
    def __init__(self, re_pattern, preproc=None, label_maps=None, unipolar=True, name=None):
        if type(re_pattern) is list:
            raise NotImplementedError("List Not Supported")

        def repm(inst):
            if preproc is not None:
                inst = preproc(inst)
            res = int(re.search(re_pattern, inst) is not None)
            if unipolar:
                resmat = [-1, 1]
            else:
                resmat = [0, 1]
            return resmat[res]

        super().__init__(exec_module=repm, label_maps=label_maps, name=name)
