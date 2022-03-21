import warnings
from abc import ABC
import random

import numpy as np
import torch
from tqdm import tqdm
import time
from torch.nn import Module
from copy import deepcopy as dc

defualt_lm_opt_cfg = {'lr': 1,
                      'epoch': 20,
                      'seed': 0,
                      'batch_size': 1024,
                      'momentum': 0.5,
                      'step_schedule': 5,
                      'step_multiplier': 1e-2}


def intercect(l1, l2):
    return [value for value in l1 if value in l2]

def union(l1, l2):
    return list(set(l1) | set(l2))

class LabelModel(ABC):
    def __init__(self, num_classes, fid2clusters, verbose=False):
        self.num_classes = num_classes
        self.fid2clusters = fid2clusters
        self.verbose = verbose
        self.num_df = len(fid2clusters)
        #########################
        # 0-indexed classes
        # Sort clusters
        #
        for fid, clusters in self.fid2clusters.items():
            crange = clusters[0]
            ccover = []
            for cluster_id, cluster in enumerate(clusters):
                cluster.sort()
                self.fid2clusters[fid][cluster_id] = cluster
                crange = intercect(crange, cluster)
                ccover = union(ccover, cluster)
            if len(crange) > 0:
                raise RuntimeError('Setup Violation: No class can appear in all groups!')
            if len(ccover) < self.num_classes:
                raise RuntimeError('Setup Violation: Class must appear at least once! Please setup a dummy label group if necessary!')
    def get_label_distribution(self, votes, batch_sz=1024):
        raise NotImplementedError('Module Not Implemented')


class GenerativeLM(LabelModel, Module, ABC):
    def __init__(self, num_classes, fid2clusters,
                 acc_prior=0.70,
                 opt_cfg=None, verbose=False,
                 opt_cb=True, device='cuda:0'):
        LabelModel.__init__(self, num_classes=num_classes,
                            fid2clusters=fid2clusters,
                            verbose=verbose)
        Module.__init__(self)
        torch.set_default_dtype(torch.float64)

        self.acc_prior = -1 * np.log(1.0 / acc_prior - 1) / 2
        self.opt_cb = opt_cb

        if opt_cfg is None:
            opt_cfg = defualt_lm_opt_cfg
        self.opt_cfg = opt_cfg

        if 'cuda' in device and torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'

        self.optimized = False

        self._init()

    def _init(self):
        raise NotImplementedError('Module Not Implemented')

    def _cll(self, votes, bid):
        raise NotImplementedError('Module Not Implemented')

    def _setup(self, votes, batch_size, shuffle=False):
        raise NotImplementedError('Module Not Implemented')

    def _norm_class_balance(self):
        return self.class_balance - torch.logsumexp(self.class_balance, dim=0)

    def _regularization(self):
        return 0.0

    def _batchize(self, votes, batch_size, shuffle=False):
        if shuffle:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :]

        batches = [
            torch.LongTensor(votes[i * batch_size: (i + 1) * batch_size, :].astype(np.int))
            for i in range(int(np.ceil(votes.shape[0] / batch_size)))
        ]

        return batches

    def optimize(self, votes, training_batch_size=None):


        self.init_random(self.opt_cfg['seed'])
        if training_batch_size is None:
            self.training_batch_size = self.opt_cfg['batch_size']
        if self.verbose:
            start = time.time()
        batches = self._setup(votes, self.training_batch_size, shuffle=True)
        if self.verbose:
            print('Setup: ', time.time() - start)

        lr = self.opt_cfg['lr']
        momentum = self.opt_cfg['momentum']
        step_size_mult = self.opt_cfg['step_multiplier']
        step_schedule = self.opt_cfg['step_schedule']
        epochs = self.opt_cfg['epoch']

        optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, nesterov=False,
            weight_decay=0)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr,
            weight_decay=0)

        if step_schedule == 'p':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-10, factor=step_size_mult)
        elif step_schedule == 'c':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-1, max_lr=0.2)
        elif step_schedule is not None and step_size_mult is not None:
            LR_milestones = list(
                filter(
                    lambda a: a > 0,
                    [i if (i % step_schedule == 0) else -1 for i in range(epochs)]
                ))

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, LR_milestones, gamma=step_size_mult)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=0.1,
                                                               last_epoch=-1)
            scheduler = None

        self.train()

        for epoch in range(epochs):
            ga = dc(self.accuracy)
            running_loss = 0.0
            if self.verbose:
                progress = tqdm(total=len(batches), desc='epoch % 3d' % (epoch + 1))
            epoch_loss = []
            for i_batch, inputs in enumerate(batches):
                optimizer.zero_grad()
                log_likelihood = self(inputs, i_batch)
                loss = -1 * torch.mean(log_likelihood)
                loss += self._regularization()
                loss.backward()
                optimizer.step()
                running_loss += loss
                epoch_loss.append(float(loss.item()))
                if self.verbose:
                    progress.set_postfix({'Train Loss: ': np.mean(epoch_loss[-3:])})
                    progress.update()
            epoch_loss = running_loss / len(batches)

            if self.verbose:
                progress.set_postfix({'Epoch Loss: ': epoch_loss.item()})
                progress.close()
            if torch.sum(torch.abs(self.accuracy - ga)) < 1e-7:
                print('1e-7 Criterion Reached: Epoch ', epoch)
                break
            if scheduler is not None:
                if step_schedule == 'p' :
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        self.optimized = True

    def forward(self, votes, bid):
        class_prior = self._norm_class_balance()
        conditional_ll = self._cll(votes, bid)
        return torch.logsumexp(class_prior + conditional_ll, dim=1)

    def get_label_distribution(self, votes, annot_batch_sz=2048):
        if not self.optimized:
            warnings.warn("Warning: Label Model not trained!")
            return None

        self.eval()
        if self.verbose:
            start = time.time()
        batches = self._setup(votes, annot_batch_sz)
        if self.verbose:
            print('Setup: ', time.time() - start)

        labels = np.ndarray((votes.shape[0], self.num_classes))
        if self.verbose:
            start = time.time()
        for batch_id, batch_votes in enumerate(batches):
            class_balance = self._norm_class_balance()
            lf_likelihood = self._cll(batch_votes, batch_id)
            jll = class_balance + lf_likelihood
            P = torch.exp(jll - torch.max(jll, dim=1)[0].unsqueeze(1).repeat(1, self.num_classes))
            P /= torch.sum(P, dim=1).unsqueeze(1).repeat(1, self.num_classes)
            labels[batch_id * annot_batch_sz:batch_id * annot_batch_sz + batch_votes.shape[0]] = P.detach().cpu().numpy()
        if self.verbose:
            print('Parallel Estimation: ', time.time() - start)
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        return labels

    def reload_fid2clusters(self, fid2clusters, oldnewfid_map):
        #TODO: Update Internal Configurations
        pass

    def get_accuracies(self):
        acc = self.accuracy.detach().cpu().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_class_balance(self):
        return np.exp(self._norm_class_balance().detach().cpu().numpy())

    def init_random(self, seed):
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available() and 'cuda' in self.device:
            torch.cuda.manual_seed_all(seed)

    def weak_label(self, votes, batch_sz=2048):
        return self.get_label_distribution(votes, annot_batch_sz=batch_sz)
