import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from AR.AR_globalNorm import AR
from LSTM.LSTM_globalNorm import LSTM
from dsanet.DSANet_globalNorm import DSANet


class run():
    def __init__(self, hyperparams,checkpoint_dir):
        self.hyperparams=hyperparams
        self.model=hyperparams.model
        self.sequence_length=hyperparams.window
        self.checkpoint_dir=checkpoint_dir
        self.checkpoint_callback=ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename='model',
                    auto_insert_metric_name=False
                )
        self.early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=True,
                    mode='min'
                )
        self.max_epochs=hyperparams.max_epochs
        self.data_name=hyperparams.data_name

    def train(self, train_loader, val_loader):

        if self.model.lower()=='dsanet':
            self.model = DSANet(self.hyperparams)
            self.onnx_file = "DSANet.onnx"
            self.model_dir = os.path.join(self.checkpoint_dir, "DSANet.ckpt")
        elif self.model.lower()=='lstm':
            self.model = LSTM(self.hyperparams)
            self.onnx_file = "LSTM.onnx"
            self.model_dir = os.path.join(self.checkpoint_dir, "LSTM.ckpt")
        elif self.model.lower()=='ar':
            self.model = AR(self.hyperparams)
            self.onnx_file = "AR.onnx"
            self.model_dir = os.path.join(self.checkpoint_dir, "AR.ckpt")

        if not os.path.exists(self.model_dir):
            trainer = Trainer(gpus=[0], default_root_dir=self.checkpoint_dir, callbacks=[self.checkpoint_callback,
                                                                                    self.early_stop]
                              , auto_lr_find=True, max_epochs=self.max_epochs)



            print("---------------------------Model training--------------------------------")
            lr_finder = trainer.tuner.lr_find(self.model, train_loader, val_loader)



            self.model.learning_rate = lr_finder.suggestion()


            trainer.fit(self.model, train_loader, val_loader)
            trainer.save_checkpoint(self.model_dir)


        trained_model = self.model.load_from_checkpoint(checkpoint_path=self.model_dir)
        trained_model.eval()
        trained_model.use_GPU = False
        self.export_onnx(self.onnx_file, trained_model)

        return trained_model

    def export_onnx(self,onnx_file, trained_model):
        x = torch.randn(1, self.sequence_length, 22)

        torch_out = trained_model(x)
        trained_model.to_onnx(onnx_file,
                              x, export_params=True)

