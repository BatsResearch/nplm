import torch


def translate_label_partitions_to_lid2fid(fid2clusters, num_classes):
    lid2fid = {}
    feature_len = len(fid2clusters)

    for lid in range(num_classes):
        lid2fid[lid] = [0] * feature_len

    # If out of bounds exception, problem with lid > range(num_classes)!
    for fid, clusters in fid2clusters.items():
        for cluster_id, cluster in enumerate(clusters):
            for lid in cluster:
                lid2fid[lid - 1][fid] = cluster_id

    return lid2fid


def isnotin(ar1, ar2):
    return torch.DoubleTensor(~(ar1[..., None] == ar2).any(-1))
