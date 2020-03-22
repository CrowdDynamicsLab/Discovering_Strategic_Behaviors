import time
import pickle
import logging
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Trainer import *


seed = 1
n_gpus = torch.cuda.device_count()
training_nlls = {'c_model':[], 'a_model':[]}


def train(trainer, rank, args):
    
    if rank==0:
        logging.info('--------------------------------------')
        logging.info("Content Batches {}, Author Batches {}".format(len(trainer.dataset.c_batches), len(trainer.dataset.a_batches)))
    
    t = time.time()
    curr_closs_sum, prev_closs_sum, curr_aloss_sum, prev_aloss_sum = 0, 0, 0, 0
    
    for i in range(1, args.max_nround+1):
        
        ###############################################
        
        prev_closs_sum = curr_closs_sum
        curr_closs_sum = 0
        
        trainer.__train_C__()
        
        for param in trainer.c_model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        trainer.c_optimizer.step()
        trainer.c_optimizer.zero_grad()
        
        dist.all_reduce(trainer.curr_closs_sum, op=dist.ReduceOp.SUM)
        curr_closs_sum = trainer.curr_closs_sum.cpu().item()
        training_nlls['c_model'].append((curr_closs_sum/trainer.dataset.c_edgesum, curr_closs_sum))
               
        all_gather_dists, all_gather_atts = \
            [torch.empty(trainer.dataset.c_max_len, args.nstrategy).float().to("cuda:{}".format(rank)) for _ in range(dist.get_world_size())], \
            [torch.empty(trainer.dataset.ca_max_len).float().to("cuda:{}".format(rank)) for _ in range(dist.get_world_size())]
        dist.all_gather(all_gather_dists, trainer.new_c_dists)
        dist.all_gather(all_gather_atts, trainer.batch_atts)
        trainer.__update_after_C__(all_gather_dists, all_gather_atts)
        del all_gather_dists, all_gather_atts
        
        ###############################################
        
        prev_aloss_sum = curr_aloss_sum
        curr_aloss_sum = 0
        
        trainer.__train_A__()
        
        for param in trainer.a_model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        trainer.a_optimizer.step()
        trainer.a_optimizer.zero_grad()
        
        dist.all_reduce(trainer.curr_aloss_sum, op=dist.ReduceOp.SUM)
        curr_aloss_sum = trainer.curr_aloss_sum.cpu().item()
        training_nlls['a_model'].append((curr_aloss_sum/trainer.dataset.a_edgesum, curr_aloss_sum))
        
          
        all_gather_dists, all_gather_atts, all_gather_alphas = \
            [torch.empty(trainer.dataset.a_max_len, args.nstrategy).float().to("cuda:{}".format(rank)) for _ in range(dist.get_world_size())], \
            [torch.empty(trainer.dataset.ac_max_len).float().to("cuda:{}".format(rank)) for _ in range(dist.get_world_size())], \
            [torch.empty(trainer.dataset.a_max_len).float().to("cuda:{}".format(rank)) for _ in range(dist.get_world_size())]
        dist.all_gather(all_gather_dists, trainer.new_a_dists)
        dist.all_gather(all_gather_atts, trainer.batch_atts)
        dist.all_gather(all_gather_alphas, trainer.batch_alphas)
        trainer.__update_after_A__(all_gather_dists, all_gather_atts, all_gather_alphas)
        del all_gather_dists, all_gather_atts, all_gather_alphas
        
        ###############################################
        
        if rank==0: logging.info('Round {}, Content NLL {:.4f}, Author NLL {:.4f}'.format(i, curr_closs_sum, curr_aloss_sum))
        if abs(curr_closs_sum-prev_closs_sum)/trainer.dataset.c_edgesum < args.threshold and abs(curr_aloss_sum-prev_aloss_sum)/trainer.dataset.a_edgesum < args.threshold and args.min_nround < i: break
      
    if rank==0: logging.info('--------------------------------------')
    
    return time.time()-t
    

def save(trainer, year, args):
    
    trainer.dataset.__update_latest__()
    pickle.dump(trainer.dataset.a_latest_dists, open(f'{args.path}/{args.strategy}_input/a_latest_{args.strategy}_dists_{year}.pkl', 'wb'), -1)
    
    pickle.dump((trainer.c_model.state_dict(), trainer.a_model.state_dict(), training_nlls), open(f'{args.path}/{args.strategy}_result/{year}_models.pkl', 'wb'), -1)
    pickle.dump((trainer.dataset.c_dist.cpu().numpy(), trainer.dataset.ca_att.cpu().numpy()), open(f'{args.path}/{args.strategy}_result/{year}_content_results.pkl', 'wb'), -1)
    pickle.dump((trainer.dataset.a_dist.cpu().numpy(), trainer.dataset.ac_att.cpu().numpy(), trainer.dataset.aa_alpha.cpu().numpy()), open(f'{args.path}/{args.strategy}_result/{year}_author_results.pkl', 'wb'), -1)
    
    
def run(year, rank, world_size, args):
    
    if rank==0:
        logging.basicConfig(level=logging.INFO,filename=f'{args.path}/run/output_{args.strategy}.log',filemode='a',format='%(asctime)s %(message)s',datefmt='%Y/%m/%d %I:%M:%S %p')
        logging.info(f'Start Year {year} with {n_gpus} GPUs')
        
    trainer = Trainer(year, args.nstrategy, args.strategy, rank, world_size, "cuda:{}".format(rank), args.path)

    if rank==0: logging.info('Prepare Input')
    trainer.__prepare__(args.c_batchsize, args.a_batchsize)
    
    if rank==0: logging.info('Start Training')
    t = train(trainer, rank, args)
    if rank==0: logging.info('Finish Training, Time {}s'.format(t))
    
    if rank==0: 
        logging.info('Save Results')
        save(trainer, year, args)
        logging.info('Finish Year {}'.format(year))
        logging.info('')
        

def init_process(rank, year, args):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    master_ip, master_port = '127.0.0.1', '12345'
    init_method = "tcp://{}:{}".format(master_ip, master_port)

    dist.init_process_group(backend='nccl', init_method=init_method, world_size=n_gpus, rank=rank)
    run(year, rank, n_gpus, args)

    
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--c_batchsize', type=int, required=True)
    parser.add_argument('--a_batchsize', type=int, required=True)
    parser.add_argument('--nstrategy', type=int, required=True)
    parser.add_argument('--strategy', type=str, required=True, choices=['cite','pub'])
    parser.add_argument('--threshold', default=2e-5, type=float)
    parser.add_argument('--max_nround', type=int, required=True)
    parser.add_argument('--min_nround', type=int, required=True)
    
    return parser.parse_args()
    

def main():
    
    args = parse_args()
    original_min_nround = args.min_nround
    for year in range(args.start_year, args.end_year):
        if year==2000: args.min_nround = int(original_min_nround/2)
        else: args.min_nround = original_min_nround           
        mp.spawn(init_process, args=(year,args), nprocs=n_gpus)
    
    
if __name__ == "__main__":
    main()    