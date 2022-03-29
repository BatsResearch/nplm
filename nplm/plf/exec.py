import numpy as np


def executor(rules, data,
             cpu_parallelism=False,
             one_indexed=True):
    '''
    :param rules: List of functions of size m
    :param data: List of instances of size n
    :return: votes: unit8 (n*m), fid2clusters
    '''

    votes = np.zeros([len(data), len(rules)])

    if cpu_parallelism:
        raise NotImplementedError("Not Yet Implemented for CPU Parallelism")
    else:
        for rule_idx, rule in enumerate(rules):
            votes[:, rule_idx] = rule.execute(data)

    fid2clusters = {}
    for rule_idx, rule in enumerate(rules):
        if rule.name is None:
            rule.set_name('Rule {:d}'.format(rule_idx))
        curr_lmap = rule.label_maps
        print(curr_lmap)
        curr_group = [0] * len(curr_lmap)
        for idx, group in curr_lmap.items():
            if one_indexed:
                curr_group[idx] = group
            else:
                curr_group[idx] = [elem + 1 for elem in group]
        fid2clusters[rule_idx] = curr_group

    return votes, fid2clusters


def evaluator(rules, gold_labels,
              pprint=False,
              class_wise_acc=False,
              average='macro'):
    ps = []
    rs = []
    accs = []
    cw_accs = []

    for rule_idx, rule in enumerate(rules):

        if class_wise_acc:
            acc, p, r, cw_acc = rule.eval(gold_labels, class_wise_acc=class_wise_acc, average=average)
            cw_accs.append(cw_acc)
        else:
            acc, p, r = rule.eval(gold_labels)
        accs.append(acc)
        ps.append(p)
        rs.append(r)

    if pprint:
        print('Name\tAcc\tP\tR')
        for idx, acc in enumerate(accs):
            print(rules[idx].name, '\t{:.4f}\t{:.4f}\t{:.4f}'.format(acc, ps[idx], rs[idx]))
    return accs, ps, rs
