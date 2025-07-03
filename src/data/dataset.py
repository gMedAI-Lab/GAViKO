from torch.utils.data import Dataset
import numpy as np
import torch
class CustomDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        path_object = self.df.loc[index]['mri_path']
        mri_file = '/workspace/data/SAG_3D_DESS_v2_full/MRI_Numpy/' + path_object
        mri_dict = np.load(mri_file)
        mri_object = mri_dict['data']

        mri_object = np.expand_dims(mri_object, 0) # (1 x 120 x 160 x 160)
        mri_object = self.transforms(mri_object)
        mri_tensor = torch.tensor(mri_object)

        label = self.df.loc[index]['kl_grade']

        return mri_tensor, label

    def __len__(self):
        return len(self.df)