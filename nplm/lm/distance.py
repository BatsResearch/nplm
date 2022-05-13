import multiprocessing
import threading
import time

import numpy as np
from scipy.spatial.distance import hamming, euclidean, cosine
from tqdm import tqdm

from .labelmodel import LabelModel
from .lm_utils import translate_label_partitions_to_lid2fid


class DistanceLM(LabelModel):
    def __init__(self, num_classes, label_partitions, offset=0, metric='hamming',
                 verbose=False, cpu_parallelism=False):
        super().__init__(num_classes, label_partitions, verbose)
        self.class_feature_map = translate_label_partitions_to_lid2fid(label_partitions, num_classes)
        self.offset = offset
        self.cpu_parallelism = cpu_parallelism and (multiprocessing.cpu_count() > 3)

        if metric == 'euclidean':
            self.metric = euclidean
        elif metric == 'cosine':
            self.metric = cosine
        else:
            self.metric = hamming

    def weak_label(self, instances):
        return self.get_label_distribution(instances)

    def optimize(self, training):
        pass

    def get_label_distribution(self, instances, batch_size=None):

        instances += self.offset

        est_labels = np.empty([len(instances)], dtype=np.int)
        default_dist = len(list(self.class_feature_map.values())[0]) + 1

        if self.verbose:
            start = time.time()
        if self.cpu_parallelism:
            def batch_decision(data_batch,
                               start_idx,
                               batch_sz):
                curr_batch_result = np.empty([batch_sz])
                for data_id, data in enumerate(data_batch):
                    min_dist = default_dist
                    curr_label = -1
                    for class_id, features in self.class_feature_map.items():
                        incorrectness = self.metric(features, data)
                        if incorrectness < min_dist:
                            min_dist = incorrectness
                            curr_label = class_id
                    curr_batch_result[data_id] = curr_label
                est_labels[start_idx:start_idx + batch_sz] = curr_batch_result

            running_threads = threading.enumerate()
            thread_num = multiprocessing.cpu_count()
            data_batches = np.array_split(instances, thread_num)
            for batch_id, batch in enumerate(data_batches):
                t = threading.Thread(target=batch_decision,
                                     args=(batch,
                                           batch_id * len(batch),
                                           len(batch),))
                t.start()

            for t in tqdm(threading.enumerate()):
                if t not in running_threads:
                    t.join()

        else:
            for data_id, data in enumerate(instances):
                min_dist = default_dist
                curr_label = -1
                for class_id, features in self.class_feature_map.items():
                    correctness = self.metric(features, data)
                    if correctness < min_dist:
                        min_dist = correctness
                        curr_label = class_id
                est_labels[data_id] = curr_label

        label_dist = np.zeros([len(instances), len(self.class_feature_map)])
        label_dist[np.arange(est_labels.size), est_labels] = 1

        if self.verbose:
            print('Total Time: ', time.time() - start)

        return label_dist
