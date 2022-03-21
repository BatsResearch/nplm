from .labelmodel import GenerativeLM
import torch
import numpy as np


class PartialLabelModel(GenerativeLM):
    def __init__(self, num_classes, fid2clusters, acc_prior=0.7,
                 opt_cfg=None, verbose=False, preset_cb=None, device='cuda:0'):
        self.preset_cb = preset_cb
        opt_cb = preset_cb is None
        if not opt_cb:
            assert len(preset_cb) == num_classes, 'Error: Incorrect shape for preset class balance: expecting {:d} get {:d}!'.format(num_classes, len(preset_cb))
        super().__init__(num_classes, fid2clusters, acc_prior, opt_cfg, verbose, opt_cb, device)

    def _init(self):
        if self.preset_cb is not None:
            self.class_balance = torch.nn.Parameter(
                torch.log(self.preset_cb),
                requires_grad=False
            )
        else:
            self.class_balance = torch.nn.Parameter(
                torch.zeros([self.num_classes], device=self.device),
                requires_grad=self.opt_cb
            )

        self.accuracy = torch.nn.Parameter(
            torch.ones([self.num_df, self.num_classes], device=self.device) * self.acc_prior,
            requires_grad=True
        )

        self.propensity = torch.nn.Parameter(
            torch.zeros([self.num_df], device=self.device),
            requires_grad=True
        )

        self.ct = torch.zeros([self.num_df, self.num_classes])
        self.poslib = torch.zeros([self.num_df, self.num_classes])
        self.neglib = torch.zeros([self.num_df, self.num_classes])
        for fid, clusters in self.fid2clusters.items():
            for cluster_id, cluster in enumerate(clusters):
                for class_id in cluster:
                    self.poslib[fid, class_id - 1] += 1
                    self.ct[fid, class_id - 1] = cluster_id
            self.neglib[fid, :] = len(clusters) - self.poslib[fid, :]
        self.poslib[self.poslib == 0] = 1


    def _setup(self, votes, batch_size, shuffle=False):
        batches = self._batchize(votes, batch_size, shuffle)
        cth = self.ct.unsqueeze(0).repeat(batch_size, 1, 1)
        self.c = torch.zeros([len(batches), batch_size, self.num_df, self.num_classes])
        self.n = torch.zeros([len(batches), batch_size, self.num_df, self.num_classes])
        self.a = torch.ones([len(batches), batch_size, self.num_df, self.num_classes])
        self.p = torch.ones([len(batches), batch_size, self.num_df])
        for bid in range(len(batches) - 1):
            extb = batches[bid].unsqueeze(2).repeat(1, 1, self.num_classes)
            self.c[bid] = torch.where(torch.eq(cth, extb), torch.tensor(1.0), torch.tensor(-1.0))
            self.a[bid] = torch.where(extb==-1, torch.tensor(0.0), torch.tensor(1.0))
            marker = torch.where(self.c[bid]==1, torch.tensor(1.0), torch.tensor(0.0))
            self.n[bid] = (1 - marker) * self.neglib + marker * self.poslib
            self.p[bid] = torch.where(batches[bid]==-1, torch.tensor(0.0), torch.tensor(1.0))

        last_bz = len(batches[-1])
        last_extb = batches[-1].unsqueeze(2).repeat(1, 1, self.num_classes)
        self.c[-1, :last_bz] = torch.where(
            torch.eq(cth[:last_bz, :, :], last_extb),
            torch.tensor(1.0), torch.tensor(-1.0))
        marker = torch.where(self.c[-1, :last_bz] == 1, torch.tensor(1.0), torch.tensor(0.0))
        self.a[-1, :last_bz] = torch.where(last_extb==-1, torch.tensor(0.0), torch.tensor(1.0))
        self.n[-1, :last_bz] = (1 - marker) * self.neglib + marker * self.poslib
        self.n = -torch.log(self.n)
        self.p[-1, :last_bz] = torch.where(batches[-1]==-1, torch.tensor(0.0), torch.tensor(1.0))
        return batches

    def _regularization(self):
        return 0.

    def _cll(self, votes, bid):
        num_inst = votes.shape[0]

        za = self.accuracy.unsqueeze(2)
        za = torch.cat((za, -1 * za), dim=2)
        za = - torch.logsumexp(za, dim=2).unsqueeze(0).repeat(num_inst, 1, 1)

        z_plh = torch.zeros((self.num_df, 1)).to(self.device)
        zp = self.propensity.unsqueeze(1)
        zp = torch.cat((zp, z_plh), dim=1)
        zp = -torch.logsumexp(zp, dim=1).unsqueeze(0).unsqueeze(-1).repeat(num_inst, 1, self.num_classes)

        cp = self.propensity.unsqueeze(0).unsqueeze(-1).repeat(num_inst, 1, self.num_classes)
        ca = self.accuracy.unsqueeze(0).repeat(num_inst,1,1)
        ab = self.a[bid][:num_inst].to(self.device)
        cc = self.c[bid][:num_inst].to(self.device)
        cn = self.n[bid][:num_inst].to(self.device)
        cll = torch.sum(((ca*cc+cn+cp+za)*ab)+zp, dim=1)

        return cll

    def get_propensities(self):
        prop = self.propensity.cpu().detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)