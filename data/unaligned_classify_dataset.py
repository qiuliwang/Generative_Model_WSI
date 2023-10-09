import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class UnalignedClassifyDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        self.dir_A_mask_cell = os.path.join(opt.maskroot, opt.phase + 'A_mask_cell')  # create a path '/path/to/data/trainA'
        self.dir_A_mask_blood = os.path.join(opt.maskroot, opt.phase + 'A_mask_blood')  # create a path '/path/to/data/trainB'

        self.dir_B_mask_cell = os.path.join(opt.maskroot, opt.phase + 'B_mask_cell')  # create a path '/path/to/data/trainA'
        self.dir_B_mask_stain = os.path.join(opt.maskroot, opt.phase + 'B_mask_stain')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_A_mask_cell = get_transform(self.opt, grayscale=True)
        self.transform_A_mask_blood = get_transform(self.opt, grayscale=True)
        self.transform_B_mask_cell = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
       
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        # print(A_path)
        # print(B_path)
        A_mask_cell_path = os.path.join(self.dir_A_mask_cell, A_path.split('/home1/qiuliwang/Data/Generative_Data/HE2CD4/trainA/')[1].split('.jpeg')[0] + '_tissues_binary.jpeg')
        A_mask_blood_path = os.path.join(self.dir_A_mask_blood, A_path.split('/home1/qiuliwang/Data/Generative_Data/HE2CD4/trainA/')[1].split('.jpeg')[0] + '.jpeg')
        A_mask_cell_img = Image.open(A_mask_cell_path)
        A_mask_blood_img = Image.open(A_mask_blood_path)

        # apply image transformation
        A = self.transform_A(A_img)
        A_mask_cell = self.transform_A_mask_cell(A_mask_cell_img)
        A_mask_blood = self.transform_A_mask_blood(A_mask_blood_img)

        '''
        CD56 data
        '''
        B_img = Image.open(B_path).convert('RGB')
        # print(self.dir_B_mask_cell)
        # /home1/qiuliwang/Data/Generative_Data/HE2CD56/trainB_mask_cell

        B_mask_cell_path = os.path.join(self.dir_B_mask_cell, B_path.split('/home1/qiuliwang/Data/Generative_Data/HE2CD4/trainB/')[1].split('.jpeg')[0] + '_tissues_binary.jpeg')
        
        B_mask_cell_img = Image.open(B_mask_cell_path)
    
        B = self.transform_B(B_img)
        B_mask_cell = self.transform_B_mask_cell(B_mask_cell_img)

        # get size label
        if 'level512' in A_path:
            A_label = np.array([1,0,0])
        elif 'level1024' in A_path:
            A_label = np.array([0,1,0])
        elif 'level2048' in A_path:
            A_label = np.array([0,0,1])

        if 'level512' in B_path:
            B_label = np.array([1,0,0])
        elif 'level1024' in B_path:
            B_label = np.array([0,1,0])
        elif 'level2048' in B_path:
            B_label = np.array([0,0,1])

        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        return {'A': A, 'A_paths': A_path, 'A_mask_cell': A_mask_cell, 'A_mask_blood': A_mask_blood, 'A_mask_cell_path': A_mask_cell_path, 'A_mask_blood_path': A_mask_blood_path, 'A_label' : A_label, 'B': B, 'B_paths': B_path, 'B_mask_cell' : B_mask_cell, 'B_mask_cell_path' : B_mask_cell_path, 'B_label' : B_label}
        
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
