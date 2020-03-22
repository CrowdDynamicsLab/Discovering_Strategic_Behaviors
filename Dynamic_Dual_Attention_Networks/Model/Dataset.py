import math
import pickle
import numpy as np

import torch


class DBLPDataset:
    
    
    def __init__(self, year, nstrategy, strategy, rank, worldsize, device, path):        
        
        self.year = year
        self.nstrategy = nstrategy
        self.strategy = strategy
        
        self.rank = rank
        self.worldsize = worldsize        
        self.device = device
        self.path = path
                
        
    def __read__(self, c_batchsize, a_batchsize):
        
        c_active, c_position, ca_adj, c_emb, c_edgellh = pickle.load(open(f'{self.path}/{self.strategy}_input/c_{self.strategy}_inputs_{self.year}.pkl', 'rb'))        
        c_edgecount = np.array([len(edge) for edge in c_edgellh])
        c_edgellh = np.array([each/sum(each) for edgellh in c_edgellh for each in edgellh]).T
        
        a_active, a_position, ac_adj, a_emb, da_emb, a_edgellh = pickle.load(open(f'{self.path}/{self.strategy}_input/a_{self.strategy}_inputs_{self.year}.pkl', 'rb'))
        a_edgecount = np.array([len(edge) for edge in a_edgellh])
        a_edgellh = np.array([each/sum(each) for edgellh in a_edgellh for each in edgellh]).T
        
        self.a_active = a_active
        self.a_position = a_position
        self.a_latest_dists = pickle.load(open(f'{self.path}/{self.strategy}_input/a_latest_{self.strategy}_dists_{self.year-1}.pkl','rb'))
        da_dist = np.array([self.a_latest_dists[a] for a in self.a_active])
        
        self.c_edgesum, self.a_edgesum = np.sum(c_edgecount), np.sum(a_edgecount)
        self.__batch__(ca_adj, ac_adj, c_batchsize, a_batchsize)
        self.__initialize__(len(c_emb), len(a_emb), ca_adj[0].shape, ac_adj[0].shape)
        self.__prepare__(ca_adj, ac_adj, c_position, a_position, c_emb, a_emb, da_emb, da_dist, c_edgecount, a_edgecount, c_edgellh, a_edgellh)
    

    def __batch__(self, ca_adj, ac_adj, c_batchsize, a_batchsize):
        
        connected_c_list, connected_c_count = np.unique(ca_adj[0],return_counts=True)
        c_batchcount = math.ceil(len(connected_c_list)/c_batchsize)
        if c_batchcount<self.worldsize:
            c_batchcount = self.worldsize
            c_batchsize = math.ceil(len(connected_c_list)/c_batchcount)        
        self.c_batches = []
        for batch_num in range(c_batchcount):
            start_arg = batch_num*c_batchsize
            end_arg = min(start_arg+c_batchsize, len(connected_c_list))
            start, end = np.sum(connected_c_count[:start_arg]), np.sum(connected_c_count[:end_arg])
            self.c_batches.append((start,end))
        
        connected_a_list, connected_a_count = np.unique(ac_adj[0],return_counts=True)
        a_batchcount = math.ceil(len(connected_a_list)/a_batchsize)
        if a_batchcount<self.worldsize:
            a_batchcount = self.worldsize
            a_batchsize = math.ceil(len(connected_a_list)/a_batchcount)
        self.a_batches = []
        for batch_num in range(a_batchcount):
            start_arg = batch_num*a_batchsize
            end_arg = min(start_arg+a_batchsize, len(connected_a_list))
            start, end = np.sum(connected_a_count[:start_arg]), np.sum(connected_a_count[:end_arg])
            self.a_batches.append((start,end))
    
    
    def __initialize__(self, nc, na, nca, nac):
        
        self.c_dist = torch.from_numpy(np.random.dirichlet([1]*self.nstrategy,size=nc)).float().to(self.device)
        self.ca_att = torch.from_numpy(np.zeros(nca, dtype=float)).float().to(self.device)
        
        self.a_dist = torch.from_numpy(np.random.dirichlet([1]*self.nstrategy,size=na)).float().to(self.device)
        self.ac_att = torch.from_numpy(np.zeros(nac, dtype=float)).float().to(self.device)
        self.aa_alpha = torch.from_numpy(np.zeros(na, dtype=float)).float().to(self.device)                
    
    
    def __prepare__(self, ca_adj, ac_adj, c_position, a_position, c_emb, a_emb, da_emb, da_dist, c_edgecount, a_edgecount, c_edgellh, a_edgellh):
        
        self.c_inputs, self.a_inputs = [], []
        self.c_edgeinfos, self.a_edgeinfos = [], []
        self.ca_unique_positions, self.ac_unique_positions = [], []
        self.ca_all_positions, self.ac_all_positions = [], []
        self.c_gather_split_infos, self.a_gather_split_infos = [[0,0] for _ in range(self.worldsize)], [[0,0] for _ in range(self.worldsize)]
    
        for batch_num, (start_pos, end_pos) in enumerate(self.c_batches):
            
            batch_adj = ca_adj[:,start_pos:end_pos]        
            c_unique, a_unique = np.sort(list(set(batch_adj[0]))), np.sort(list(set(batch_adj[1])))
            c_unique_position, a_unique_position = np.array([c_position[c] for c in c_unique]), np.array([a_position[a] for a in a_unique])
            self.ca_all_positions.append([c_unique_position, a_unique_position])
            self.c_gather_split_infos[batch_num%self.worldsize][0] += len(c_unique_position)
            self.c_gather_split_infos[batch_num%self.worldsize][1] += (end_pos-start_pos)
            
            if batch_num%self.worldsize!=self.rank: continue
            self.ca_unique_positions.append([c_unique_position, a_unique_position])
            
            batch_c_emb, batch_a_emb = torch.from_numpy(c_emb[c_unique_position]).float().to(self.device), torch.from_numpy(a_emb[a_unique_position]).float().to(self.device)
            batch_a_dist = self.a_dist[a_unique_position]
            batch_trans_adj = np.array([[np.argwhere(c_unique==c)[0,0] for c in batch_adj[0]], \
                               [np.argwhere(a_unique==a)[0,0] for a in batch_adj[1]]])            
            self.c_inputs.append([batch_c_emb, batch_a_emb, batch_trans_adj, batch_a_dist])
            
            batch_edgecount = c_edgecount[c_unique_position[0]:c_unique_position[-1]+1]
            batch_edgellh = torch.from_numpy(c_edgellh[:,np.sum(c_edgecount[:c_unique_position[0]]):np.sum(c_edgecount[:c_unique_position[-1]+1])]).float().to(self.device)
            self.c_edgeinfos.append([batch_edgecount, batch_edgellh])                    
        
        self.c_max_len, self.ca_max_len = max([self.c_gather_split_infos[_rank][0] for _rank in range(self.worldsize)]), max([self.c_gather_split_infos[_rank][1] for _rank in range(self.worldsize)])
        
        
        for batch_num, (start_pos, end_pos) in enumerate(self.a_batches):
            
            batch_adj = ac_adj[:,start_pos:end_pos]
            a_unique, c_unique = np.sort(list(set(batch_adj[0]))), np.sort(list(set(batch_adj[1])))
            a_unique_position, c_unique_position = np.array([a_position[a] for a in a_unique]), np.array([c_position[c] for c in c_unique])
            self.ac_all_positions.append([a_unique_position, c_unique_position])
            self.a_gather_split_infos[batch_num%self.worldsize][0] += len(a_unique_position)
            self.a_gather_split_infos[batch_num%self.worldsize][1] += (end_pos-start_pos)
            
            if batch_num%self.worldsize!=self.rank: continue
            self.ac_unique_positions.append([a_unique_position, c_unique_position])
            
            batch_a_emb, batch_c_emb, batch_da_emb = torch.from_numpy(a_emb[a_unique_position]).float().to(self.device), torch.from_numpy(c_emb[c_unique_position]).float().to(self.device), torch.from_numpy(da_emb[a_unique_position]).float().to(self.device)  
            batch_c_dist, batch_da_dist = self.c_dist[c_unique_position], torch.from_numpy(da_dist[a_unique_position]).float().to(self.device)
            batch_trans_adj = np.array([[np.argwhere(a_unique==a)[0,0] for a in batch_adj[0]], \
                               [np.argwhere(c_unique==c)[0,0] for c in batch_adj[1]]])
            self.a_inputs.append([batch_a_emb, batch_c_emb, batch_da_emb, batch_da_dist, batch_trans_adj, batch_c_dist])            
            
            batch_edgecount = a_edgecount[a_unique_position[0]:a_unique_position[-1]+1]
            batch_edgellh = torch.from_numpy(a_edgellh[:,np.sum(a_edgecount[:a_unique_position[0]]):np.sum(a_edgecount[:a_unique_position[-1]+1])]).float().to(self.device)
            self.a_edgeinfos.append([batch_edgecount, batch_edgellh])
            
        self.a_max_len, self.ac_max_len = max([self.a_gather_split_infos[_rank][0] for _rank in range(self.worldsize)]), max([self.a_gather_split_infos[_rank][1] for _rank in range(self.worldsize)])
                            
    
    def __update_after_A__(self, all_gather_dists, all_gather_atts, all_gather_alphas):
        
        for rank, (dists, atts, alphas) in enumerate(zip(all_gather_dists, all_gather_atts, all_gather_alphas)):
            
            num = 0
            a_start, a_end, ac_start, ac_end = 0, 0, 0, 0
            while num*self.worldsize+rank < len(self.a_batches):
                
                a_unique = self.ac_all_positions[num*self.worldsize+rank][0]
                start_pos, end_pos = self.a_batches[num*self.worldsize+rank]
                a_end = a_start + len(a_unique)
                ac_end = ac_start + (end_pos-start_pos)
                
                self.a_dist[a_unique] = dists[a_start:a_end]
                self.aa_alpha[a_unique] = alphas[a_start:a_end]
                self.ac_att[start_pos:end_pos] = atts[ac_start:ac_end]
                
                num += 1
                a_start, ac_start = a_end, ac_end                  
        
        for batch_num, unique_position in enumerate(self.ca_unique_positions):
            self.c_inputs[batch_num][-1] = self.a_dist[unique_position[1]]
            
            
    def __update_after_C__(self, all_gather_dists, all_gather_atts):
        
        for rank, (dists, atts) in enumerate(zip(all_gather_dists, all_gather_atts)):
            
            num = 0
            c_start, c_end, ca_start, ca_end = 0, 0, 0, 0
            while num*self.worldsize+rank < len(self.c_batches):
                
                c_unique = self.ca_all_positions[num*self.worldsize+rank][0]
                start_pos, end_pos = self.c_batches[num*self.worldsize+rank]
                c_end = c_start + len(c_unique)
                ca_end = ca_start + (end_pos-start_pos)
                
                self.c_dist[c_unique] = dists[c_start:c_end]
                self.ca_att[start_pos:end_pos] = atts[ca_start:ca_end]
                
                num += 1
                c_start, ca_start = c_end, ca_end  
        
        for batch_num, unique_position in enumerate(self.ac_unique_positions):
            self.a_inputs[batch_num][-1] = self.c_dist[unique_position[1]]
            
            
    def __update_latest__(self):
        
        a_dist_cpu = self.a_dist.cpu().numpy()
        for a in self.a_active:
            self.a_latest_dists[a] = a_dist_cpu[self.a_position[a]]