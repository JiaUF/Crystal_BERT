from .optim_scheduler import ScheduledOptim
from  Crystal_BERT.BERT import BERT
from torch.utils.data import DataLoader
import torch
from Crystal_BERT.MLM import Crystal_BERT
import torch.nn as nn
import tqdm as tqdm

class Trainer():
    """
    Training class for Crystal_BERT
    1: Atomic Position MLM Trainer
    2: Atomic Number MLM Trainer
    """
    def __init__(self, bert: BERT,
                 train_data: DataLoader, test_data: DataLoader = None,
                 lr = 0.04, betas = (0.9, 0.999), weight_decay = 0.01, warmup_step = 10000,
                 with_cuda: bool = True, cuda_devices = None, log_freq = 10):
        """
        Param bert: Model to be trained
        Param train_data: training data from dataloader
        Param test_data: testing data from dataloader
        Param lr: learning rate
        Param beta: Adam optimizer beta
        Param weight_decay: Adam optimizer decay
        Param cuda: training with cuda
        Param log_freq: logging frequency for batch
        """
    
        #setting device for trainin
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        #bert model to be saved
        self.bert = bert

        #masked language models
        self.model = Crystal_BERT(self.bert).to(self.device)

        #multiple GPU training
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            
        #training and test data
        self.train_data = train_data
        self.test_data = test_data

        #optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)
        self.scheduled_optimizer = ScheduledOptim(self.optimizer, self.bert.d_model, n_warmup_steps=warmup_step)
        
        #loss functions for 
        self.atomic_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.position_loss = nn.MSELoss()
        
        #log frequency
        self.log_freq = log_freq
        
        #training_metrics
        self.train_metrics = {"train_total_loss": [],
                              "train_atomic_loss": [],
                              "train_position_loss": []
                             }
        
        self.test_metrics = {"test_total_loss": [],
                              "test_atomic_loss": [],
                              "test_position_loss": []
                            }
                              
        print("Total Parameters:", sum([l.nelement() for l in self.model.parameters()]))
        
    def train(self, epoch):
        self.iteration(epoch, self.train_data)
    
    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
    
    def iteration(self, epoch, data_loader, train=True):
        """loops over data loader for training
        Param epoch: number of epoch to train
        Param data_loader: dataloader to loop over
        Param train: boolean for test or train
        """
        str_code = "train" if train else "test"
        
        #tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{bar:20}{r_bar}")
        avg_loss = 0.0
        for i, data in data_iter:
            #send batch to gpu or cpu
            data = {key: value.to(self.device) for key, value in data.items()}
            
            #ouput
            atom, position = self.model.forward(bert_input = data["bert_input"].type(torch.FloatTensor), mask = data["attn_mask"])
            
            #computes loss
            atom_loss = self.atomic_loss(atom.transpose(1,2), data["atom"][:, :, 0].type(torch.LongTensor))
            position_loss = self.position_loss(position, data["atom"][:, :, 1:4].type(torch.FloatTensor))
            
            total_loss = atom_loss + position_loss
            
            if not train:
                self.test_metrics["test_total_loss"].append(total_loss.item())
                self.test_metrics["test_atomic_loss"].append(atom_loss.item())
                self.test_metrics["test_position_loss"].append(position_loss.item())
            
            if train:
                self.scheduled_optimizer.zero_grad()
                total_loss.backward()
                self.scheduled_optimizer.step_and_update_lr()
                
                self.train_metrics["train_total_loss"].append(total_loss.item())
                self.train_metrics["train_atomic_loss"].append(atom_loss.item())
                self.train_metrics["train_position_loss"].append(position_loss.item())
            
            avg_loss += total_loss.item()
            
            #post
            post_fix = {"epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i+1)
                       }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
            
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))          
                         
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        torch.save(self.model.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    def save_metrics(self, train_metric_dir, test_metric_dir):
        np.save(train_metric_dir, self.train_metrics)
        np.save(test_metric_dir, self.test_metrics)