import os

import torch
import argparse
from Trainer.Trainer import Trainer
from Crystal_BERT.BERT import BERT
from Dataset import Atoms_Dataset
from torch.utils.data import DataLoader
import warnings 
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #directory
    parser.add_argument("-path", "--path", required = True, type = str, help = "path to training data")
    parser.add_argument("--s", required = False, type = tuple, default = (0.8, 0.2), help = "train-test split")
    parser.add_argument("--output_dir", required = False, type = str, help = "output directory of model")
    
    #model logistics
    parser.add_argument("-d_model", "--d_model", default = 120, help = "dimension of the model")
    parser.add_argument("-d_ff", "--d_ff", default = 3000, help = "feed foward size")
    parser.add_argument("-max_len", "--max_len", default = 50, help = "max length for padding")
    parser.add_argument("-num_layers", "--num_layers", default = 12, help = "number of transformer layers")
    parser.add_argument("-num_heads", "--num_heads", default = 12, help = "number of attention heads")

    #training parameters
    parser.add_argument("-epoch", "--epoch", required = False, type = int, default = 1000, help = "epoch to train")
    parser.add_argument("-split", "--split", required = False, type = float, default = 0.8, help = "percentage of dataset for training")
    parser.add_argument("-batch_size", "--batch_size", required = True, type = int, default = 60, help = "batch size for training")
    parser.add_argument("-num_workers", "--num_workers", type = int, default = 5, help = "number of workers for dataloader")
    parser.add_argument("-random_transform", "--random_transform", type = bool, default = False,  help = "implementing random transform")
    parser.add_argument("-shuffle", "--shuffle", type = bool, default = True, help = "shuffle for dataloader")
    
    #training logistics
    parser.add_argument("-with_cuda", "--with_cuda", type = bool, default = True, help = "training on CUDA T/F")
    parser.add_argument("-log_freq", "--log_freq", type = int, default = 10, help = "log frequency")
    parser.add_argument("-on_memory", "--on_memory", type = bool, default = True, help = "loading ono memory T/F")
    parser.add_argument("-cuda_devices", "--cuda_devices", type = int, nargs = "+", default = None, help = "CUDA devices")
    
    parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate of Adam")
    parser.add_argument("--beta1", type = float, default = 0.9, help = "adam first beta value")
    parser.add_argument("--beta2", type = float, default = 0.999, help = "adam second beta value")
    parser.add_argument("--weight_decay", type = float, default = 0.1, help = "weight decay of atom")
    
    args = parser.parse_args()
    
    os.makedirs("output/bert_trained", exist_ok = True)
    output_dir = "output/bert_trained"
        
    os.makedirs("output/train_metric", exist_ok = True)
    os.makedirs("output/test_metric", exist_ok = True)
    train_dir = "output/train_metric"
    test_dir = "output/test_metric"
    
    print("Loading training data")
    config = {"path": args.path,
              "random_transform_args": args.random_transform,
              "max_len": args.max_len
             }
    
    dataset = Atoms_Dataset(config)
    
    print("Splitting into train and test sets")
    train_set, test_set = torch.utils.data.random_split(dataset, [args.split, 1 - args.split])
    
    print("Initializing dataloaders")
    train_dataloader = DataLoader(train_set, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = args.num_workers)
    test_dataloader = DataLoader(test_set, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = args.num_workers)
    
    print("Loading BERT")
    bert = BERT(heads = args.num_heads, d_model = args.d_model, hidden = args.d_ff, n_layers = args.num_layers)
    print("Initializing trainer")
    pre_train = Trainer(bert, train_dataloader, test_dataloader, lr = args.lr, 
                        betas = (args.beta1, args.beta2), weight_decay = args.weight_decay, 
                        with_cuda = args.with_cuda, cuda_devices = args.cuda_devices, log_freq = args.log_freq)
    
    print("Training start")
    for epoch in range(args.epoch):
        pre_train.train(epoch)
        pre_train.save(epoch, output_dir)
        
        if test_set is not None:
            pre_train.test(epoch)
    pre_train.save_metrics(train_dir, test_dir)