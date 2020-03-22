import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicAttentionLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, alpha, device):
        super(DynamicAttentionLayer, self).__init__()
        
        self.alpha = alpha
        self.device = device
        
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dW = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.dW.data, gain=1.414)
        self.da = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.da.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, a_feats, b_feats, da_feats, b_dis, da_dis, adj_ab):
        
        ha_feats = torch.mm(a_feats, self.W)
        hb_feats = torch.mm(b_feats, self.W)        
        compare = torch.cat([ha_feats[adj_ab[0]], hb_feats[adj_ab[1]]], dim=1)
        e = self.leakyrelu(torch.matmul(compare, self.a).squeeze(1))        
        attention = -9e15*torch.ones(len(a_feats), len(b_feats)).to(self.device)
        attention[adj_ab[0],adj_ab[1]] = e
        attention = F.softmax(attention, dim=1)
        
        ga_feats = torch.mm(a_feats, self.dW)
        gda_feats = torch.mm(da_feats, self.dW)
        dcompare = torch.cat([ga_feats, gda_feats], dim=1)
        de = torch.matmul(dcompare, self.da).squeeze(1)
        alpha = torch.sigmoid(de)
        
        new_a_dis = (1-alpha).reshape(-1,1)*torch.matmul(attention, b_dis) + alpha.reshape(-1,1)*da_dis
        new_a_dis = torch.tanh(new_a_dis)
        new_a_dis = F.normalize(new_a_dis,p=1,dim=-1)
        
        return new_a_dis, attention.detach(), alpha.detach()
    

class DynamicAttentionModule(nn.Module):
    
    def __init__(self, in_dim, out_dim, alpha, device):
        super(DynamicAttentionModule, self).__init__()
        
        self.att_layer = DynamicAttentionLayer(in_dim, out_dim, alpha, device).to(device) 
        
    def forward(self, batch_input):
        
        batch_a_feats, batch_b_feats, batch_da_feats, batch_da_dis, batch_trans_adj, batch_b_dis = batch_input

        new_a_dis, batch_att, alpha = self.att_layer(batch_a_feats, batch_b_feats, batch_da_feats, batch_b_dis, batch_da_dis, batch_trans_adj)
        
        batch_att = batch_att[batch_trans_adj[0], batch_trans_adj[1]]        
        return new_a_dis, batch_att, alpha