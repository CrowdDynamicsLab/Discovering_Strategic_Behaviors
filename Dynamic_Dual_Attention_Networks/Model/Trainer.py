import numpy as np

import torch
import torch.optim as optim

from Dataset import *
from StaticAttention import *
from DynamicAttention import *


nfeat = 50
alpha = 0.2
c_lr, a_lr = 0.001, 0.005
weight_decay = 5e-4


class Trainer:

    
    def __init__(self, year, nstrategy, strategy, rank, worldsize, device, path):  
        
        self.dataset = DBLPDataset(year, nstrategy, strategy, rank, worldsize, device, path)
        
        self.c_model = StaticAttentionModule(in_dim=nfeat, out_dim=int(nfeat/2), alpha=alpha, device=device)
        self.c_optimizer = optim.Adam(self.c_model.parameters(), lr=c_lr, weight_decay=weight_decay)
        self.a_model = DynamicAttentionModule(in_dim=nfeat, out_dim=int(nfeat/2), alpha=alpha, device=device)
        self.a_optimizer = optim.Adam(self.a_model.parameters(), lr=a_lr, weight_decay=weight_decay)
        
        self.nstrategy = nstrategy
        self.device = device
        self.curr_closs_sum, self.curr_aloss_sum = torch.tensor([0]).float().to(device), torch.tensor([0]).float().to(device)
        
    
    def __prepare__(self, c_batchsize, a_batchsize):
        
        self.dataset.__read__(c_batchsize, a_batchsize)
        
    
    def __loss__(self, batch_edge, new_dist, batch_edgellh):
        
        batch_edgedist = []
        for node_index, count in enumerate(batch_edge):
            for _ in range(count):
                batch_edgedist.append(new_dist[node_index])
        batch_edgedist = torch.stack(batch_edgedist, dim=0)
        loss = -torch.sum(torch.log(torch.bmm(batch_edgedist[:,None,:],batch_edgellh.t()[:,:,None]).squeeze()))/len(batch_edgedist)
        
        return loss
    
    
    def __train_A__(self):
        
        self.a_model.train()        
        self.curr_aloss_sum[0] = 0
        self.new_a_dists, self.batch_atts, self.batch_alphas = [], [], []
        
        for batch_num, batch in enumerate(self.dataset.a_inputs):
            
            new_a_dist, batch_att, batch_alpha = self.a_model(batch)
            self.new_a_dists.append(new_a_dist)
            self.batch_atts.append(batch_att)
            self.batch_alphas.append(batch_alpha)
            
            a_loss = self.__loss__(self.dataset.a_edgeinfos[batch_num][0], new_a_dist, self.dataset.a_edgeinfos[batch_num][1])
            a_loss.backward()
            self.curr_aloss_sum += (a_loss.detach()*np.sum(self.dataset.a_edgeinfos[batch_num][0]))
            
        self.new_a_dists, self.batch_atts, self.batch_alphas = torch.cat(self.new_a_dists), torch.cat(self.batch_atts), torch.cat(self.batch_alphas)
        self.new_a_dists = torch.cat([self.new_a_dists, torch.zeros(self.dataset.a_max_len-len(self.new_a_dists), self.nstrategy).float().to(self.device)])
        self.batch_atts = torch.cat([self.batch_atts, torch.zeros(self.dataset.ac_max_len-len(self.batch_atts)).float().to(self.device)])
        self.batch_alphas = torch.cat([self.batch_alphas, torch.zeros(self.dataset.a_max_len-len(self.batch_alphas)).float().to(self.device)])
    
    
    def __train_C__(self):
        
        self.c_model.train()        
        self.curr_closs_sum[0] = 0
        self.new_c_dists, self.batch_atts = [], []
        
        for batch_num, batch in enumerate(self.dataset.c_inputs):
            
            new_c_dist, batch_att = self.c_model(batch)
            self.new_c_dists.append(new_c_dist)
            self.batch_atts.append(batch_att)
            
            c_loss = self.__loss__(self.dataset.c_edgeinfos[batch_num][0], new_c_dist, self.dataset.c_edgeinfos[batch_num][1])
            c_loss.backward()
            self.curr_closs_sum += (c_loss.detach()*np.sum(self.dataset.c_edgeinfos[batch_num][0]))
            
        self.new_c_dists, self.batch_atts = torch.cat(self.new_c_dists), torch.cat(self.batch_atts)
        self.new_c_dists = torch.cat([self.new_c_dists, torch.zeros(self.dataset.c_max_len-len(self.new_c_dists), self.nstrategy).float().to(self.device)])
        self.batch_atts = torch.cat([self.batch_atts, torch.zeros(self.dataset.ca_max_len-len(self.batch_atts)).float().to(self.device)])

        
    def __update_after_A__(self, all_gather_dists, all_gather_atts, all_gather_alphas):
        
        self.dataset.__update_after_A__(all_gather_dists, all_gather_atts, all_gather_alphas)
        
        
    def __update_after_C__(self, all_gather_dists, all_gather_atts):
        
        self.dataset.__update_after_C__(all_gather_dists, all_gather_atts)