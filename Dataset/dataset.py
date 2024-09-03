import numpy as np
from torch import tensor
from torch.utils.data import Dataset
import ase
from ase import io
import os
import torch 
import random
import torch.nn.functional as Func


class Atoms_Dataset(Dataset):
    """
    Dataset class using ase.atoms object:
    args:
        config(dict):
            path: directory of path of cif files
            random_transform_args: (Number of random transforms performed) #dont implement this now
            transform: arg for transforming data (Not implemented yet)
    """
    def __init__(
        self, 
        config,
        transform = False, 
        random_transform = None,
        mask_rate = 0.15
    ):
        self.config = config
        self.cif_path = self.config["path"]
        self.cifs = os.listdir(self.cif_path)
        self.max_len = config["max_len"]
        self.mask_rate = mask_rate
        
        self.random_transform = random_transform
        self.transform = transform
        
    def __len__(self):
        return len(self.cifs)
    
    def __getitem__(self, idx):
        temp_id = os.path.join(self.cif_path, self.cifs[idx])
        atoms = self.get_atoms(temp_id)
        original = self.padding(tensor(np.column_stack((atoms.get_atomic_numbers(), atoms.get_positions()))))
        min_distance = min((atoms.get_all_distances()[atoms.get_all_distances()!=0]).flatten())
            
        t = {"og": np.column_stack((atoms.get_atomic_numbers(), atoms.get_positions())),
             "trans": np.column_stack((atoms.get_atomic_numbers(), atoms.get_positions()))
            }
        corrupted, output_labels = self.random_masking(tensor(t["trans"]), min_distance = min_distance)
        bert_input = self.padding(corrupted)
        bert_labels = self.padding(output_labels)
        attn_mask = self.attn_mask(bert_input)
        output = {"atom": original,
                  "bert_input": bert_input,
                  "bert_labels": bert_labels,
                  "attn_mask": attn_mask
                 }
        
        return {key: torch.tensor(value) for key, value in output.items()}
    
    def get_atoms(self, idx):
        atoms = io.read(idx)
        return atoms
    
    def padding(self, array):
        #pad array to max len with 0
        return Func.pad(array, (0,0,0,self.max_len - len(array)))
    
    def random_masking(self, array, min_distance, mask_rate=0.15):
        #mask 15% of the time. When masked:
        #80% of tokens changed to masking tokens (119, 119, 119, 119)
        #10% of tokens changed to randomly selected tokens (atomic_num, rand_x, rand_y, rand_z)
        #10% of tokens remain the same
        corrupt = array.clone().detach()
        output_labels = torch.zeros(corrupt.shape)

        for i, atom in enumerate(corrupt):
            prob = random.random()
            if prob < mask_rate:
                prob /= mask_rate
                if prob < 0.8: 
                    output_labels[i] = (corrupt[i])
                    corrupt[i] = torch.cat((tensor(119).reshape(1), tensor(corrupt[i][1:4] + np.random.randn(3) * min_distance).reshape(3)))

                elif prob < 0.9:
                    output_labels[i] = (corrupt[i])
                    corrupt[i] = torch.cat((torch.randint(1, 118, size = (1,)).reshape(1), tensor(corrupt[i][1:4] + np.random.randn(3) * min_distance).reshape(3)))
            else:
                output_labels[i] = (tensor([0,0,0,0]))
        return corrupt, output_labels
    
    def attn_mask(self, array):
        return array[:,0] > 0