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



class LSTM(pl.LightningModule):
    def __init__(self, hparams):
        super(LSTM, self).__init__()
        self.batch_size = hparams.batch_size
        self.n_layers = hparams.n_layers
        self.n_multiv = hparams.n_multiv
        self.n_out_multiv = hparams.n_out_multiv
        self.output_length=hparams.output_length
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

        self.lstm = nn.LSTM(self.n_multiv, self.hidden_size, self.n_layers, batch_first=True)
        self.linear_1 = nn.Linear(self.hidden_size, self.n_out_multiv)
        self.linear_2 = nn.Linear(self.window, self.output_length)
        #self.batchNorm = nn.BatchNorm1d(36, affine=True)

    def forward(self, x):

        self.h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        self.c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)

        if self.use_GPU:
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

        lstm_out, (hn, cn) = self.lstm(x, (self.h0.detach(), self.c0.detach()))
        output = self.linear_1(lstm_out)
        output = torch.transpose(output, 1, 2)
        predictions=self.linear_2(output)
        predictions=torch.transpose(predictions, 1,2)


        return predictions

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def loss(self, labels, predictions):

        if self.criterion == 'l1_loss':
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == 'mse_loss':
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
        write.writerow(['LSTM', MSE_average, RMSE_average, MAE_average])

