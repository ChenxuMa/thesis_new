import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import data_processing_globalNorm as proc
from data_processing_globalNorm import data_process
class Data(pl.LightningDataModule):
    def __init__(self, hparams, data_file_path):

        super(Data, self).__init__()
        self.batch_size=hparams.batch_size
        self.window = hparams.window
        self.n_multiv = hparams.n_multiv
        self.output_length=hparams.output_length
        self.dataset_path=hparams.dataset_path

        self.train_list=[]
        self.val_list=[]
        self.test_list=[]
        self.data_processing = proc.data_process(self.dataset_path, self.window,self.output_length
                                            , data_file_path, train_test_percentage=0.8,
                                                 validation_percentage=0.1)
        #self.label_scaler_list=self.data_processing.scaler_dict
    def prepare_data(self):


        # # self.scaler_dict=data_processing.scaler_dict
        # for car, data in self.data_processing.traindict.items():
        #     self.train_list.extend(data)
        # for car, data in self.data_processing.val_dict.items():
        #     self.val_list.extend(data)
        # for car, data in self.data_processing.testdict.items():
        #     self.test_list.extend(data)
        self.train_dataloader()
        self.val_dataloader()
        self.test_dataloader()
    def train_dataloader(self):
        return DataLoader(self.data_processing.train_list, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def val_dataloader(self):

        return DataLoader(self.data_processing.val_list, batch_size=self.batch_size, shuffle=False, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.data_processing.test_list, batch_size=self.batch_size, shuffle=False, drop_last=True)
        # return self.data_processing.testdict
    def get_scaler_dict(self):
        return self.data_processing.scaler_dict
    def get_mean_dict(self):
        return self.data_processing.mean_dict
    def get_std_dict(self):
        return self.data_processing.std_dict