import os
import ase
from ase import io
import argparse
import copy 
import numpy as np

def random_rotation(atoms):
    """
    Apply random rotation to ase.atoms object
    atoms_object --> rotated atoms_object
    """
    atoms_copy = copy.deepcopy(atoms)
    angle = 360*np.random.random()
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    atoms_copy.rotate(angle, v=axis, rotate_cell=False)
    return atoms_copy

def random_translation(atoms):
    """
    Apply random translation to ase.atoms object (fractional coords)
    atoms_onject --> translated atoms_object
    """
    atoms_copy = copy.deepcopy(atoms)
    fractional_translation = np.random.rand(3)
    real_translation = np.dot(atoms.cell, fractional_translation)
    atoms_copy.translate(real_translation)
    atoms_copy.wrap()
    return atoms_copy

def random_trans(atoms):
    """
    Applies random transform N times randomly
    """
    rand = np.random.randint(0,1)
    if rand == 0:
        trans = random_rotation(atoms)
    else:
        trans = random_translation(atoms)
    return trans

class Create_Augmented_Set:
    """
    Creates a set of augmented data:
    params:
    path: path of cif files of original dataset
    output_path: output path of augmented data
    mode: mode for performing augmentations
        0: translations only
        1: rotations only
        2: combination of both
    """
    def __init__(self, path: str, output_path: str, mode = 0):
        self.path = path
        self.output_path = output_path
        self.mode = mode
        self.path_dir = os.listdir(path)
        
    def trans(self, atom):
        #takes in atom and performs base on the mode
        if self.mode == 0:
            transformed = random_translation(atom)
        if self.mode == 1:
            transformed = random_rotation(atom)
        if self.mode == 2:
            transformed = random_trans(atom)
        return transformed
    
    def read_cif(self, file: str):
        #reads in the cif and stores as atoms
        return ase.io.read(file)
        
    def aug_dataset(self, N):
        """
        Creates a new set with augmented structures. 
        input:
        N: number of augmentation per structure
        output:
        set of augmented structures at output path
        """
        for files in self.path_dir:
            if files.endswith(".cif"):
                remove_cif = files.removesuffix('.cif')
                atom = self.read_cif(os.path.join(self.path, files))
                for i in range(N):
                    aug_str = remove_cif + "aug_" + str(i+1) + ".cif"
                    transformed = self.trans(atom)
                    ase.io.write(os.path.join(self.output_path, aug_str), transformed)
            else:
                break
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", metavar= "N", type = int, help = "Number of augmentations to be performed")
    parser.add_argument("--path", metavar = "path", type = str, help = "Path of dataset")
    parser.add_argument("--output_path", metavar = "output", type = str, default = None, help = "Output path of data augmentation")
    
    namespace = parser.parse_args()
    
    N = namespace.N
    path = namespace.path
    output = namespace.output_path
    
    if output != None:
        z = Create_Augmented_Set(path, output)
        z.aug_dataset(N)
    
    os.makedirs(os.path.join(path, "augmented"), exist_ok=True)
    z = Create_Augmented_Set(path, os.path.join(path, "augmented"))
    z.aug_dataset(N)