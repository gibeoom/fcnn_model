import pandas as pd
import numpy as np
import os

np.random.seed(999)

class MaruDataset:

    def __init__(self, basepath, *, val_dataset_size):
        self.basepath = basepath
        self.val_dataset_size = val_dataset_size

    def _get_maru_filenames(self, male_dataframe_name='male_files_train.txt', female_dataframe_name='female_files_train.txt'):
        male_metadata = pd.read_csv(os.path.join(self.basepath, male_dataframe_name))
        female_metadata = pd.read_csv(os.path.join(self.basepath, female_dataframe_name))
        male_clean_files = np.transpose(male_metadata).values[0]
        female_clean_files = np.transpose(female_metadata).values[0]
        np.random.shuffle(male_clean_files)
        np.random.shuffle(female_clean_files)
        print("Total number of male training examples:", len(male_clean_files))
        print("Total number of female training examples:", len(female_clean_files))
        return male_clean_files, female_clean_files

    def get_train_val_filenames(self):
        male_clean_files, female_clean_files = self._get_maru_filenames(male_dataframe_name='male_files_train.txt', female_dataframe_name='female_files_train.txt')

        # # resolve full path
        # clean_files = [os.path.join(self.basepath, 'clips', 'train', filename) for filename in clean_files]

        male_clean_files = male_clean_files[:-(self.val_dataset_size//2)]
        male_clean_val_files = male_clean_files[-(self.val_dataset_size//2):]
        female_clean_files = female_clean_files[:-(self.val_dataset_size // 2)]
        female_clean_val_files = female_clean_files[-(self.val_dataset_size // 2):]

        clean_files = np.concatenate((male_clean_files, female_clean_files), axis=0)
        clean_val_files = np.concatenate((male_clean_val_files, female_clean_val_files), axis=0)
        np.random.shuffle(clean_files)
        np.random.shuffle(clean_val_files)
        print("# of Training clean files:", len(clean_files))
        print("# of  Validation clean files:", len(clean_val_files))
        return clean_files, clean_val_files


    def get_test_filenames(self):
        male_clean_files, female_clean_files = self._get_maru_filenames(male_dataframe_name='male_files_test.txt', female_dataframe_name='female_files_test.txt')

        # # resolve full path
        # clean_files = [os.path.join(self.basepath, 'clips', 'test', filename) for filename in clean_files]
        clean_files = np.concatenate((male_clean_files, female_clean_files), axis=0)

        print("# of Testing clean files:", len(clean_files))
        return clean_files