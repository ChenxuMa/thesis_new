import csv
import math

import torch.nn as nn
import torch
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from data_globalNorm import Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from numpy import *


class AR(pl.LightningModule):
    def __init__(self, hparams):
        super(AR, self).__init__()
        self.batch_size = hparams.batch_size
        self.n_layers = hparams.n_layers
        self.n_multiv = hparams.n_multiv
        self.n_out_multiv = hparams.n_out_multiv
        self.hidden_size = hparams.n_hidden
        self.window = hparams.window
        self.learning_rate = hparams.lr
        self.criterion = hparams.criterion
        self.__build_model()
        self.use_GPU = hparams.use_GPU
        self.save_hyperparameters()

    def __build_model(self):
        print("----------------------build model-------------------------")
        """
        Layout model
        """


        self.linear_1 = nn.Linear(self.window, 40)
        self.linear_2 = nn.Linear(self.n_multiv, 1)

        print("----------------------complete----------------------------")



    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.linear_2(x)
        return x

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def loss(self, labels, predictions):

        if self.criterion == 'l1_loss':

            loss = F.l1_loss(predictions, labels)
        elif self.criterion == 'mse_loss':
            # print(predictions.size())
            # print(labels.size())
            loss = F.mse_loss(predictions, labels)

        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)
        return loss_val

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        self.log("val_loss", loss_val)
        return loss_val

    def test_model(self, test_loader, result_file_name):

        MSE_loss = []
        RMSE_loss = []
        MAE_loss = []
        print("---------------------------testing model-----------------------------------")
        for id, (test_input, test_label) in enumerate(test_loader):
            output = self.forward(test_input)

            MSE_loss_result = F.mse_loss(output, test_label).detach().numpy().tolist()
            RMSE_loss_result = math.sqrt(MSE_loss_result)
            MAE_loss_result = F.l1_loss(output, test_label).detach().numpy().tolist()
            MSE_loss.append(MSE_loss_result)
            RMSE_loss.append(RMSE_loss_result)
            MAE_loss.append(MAE_loss_result)

        MSE_average = mean(MSE_loss)
        RMSE_average = mean(RMSE_loss)
        MAE_average = mean(MAE_loss)
        header = ['model', 'MSE_loss', 'RMSE_loss', 'MAE_loss']
        file = open(result_file_name, 'a', newline='')
        write = csv.writer(file)
        write.writerow(header)
        write.writerow(['AR', MSE_average, RMSE_average, MAE_average])

