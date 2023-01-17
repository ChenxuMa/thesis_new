import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import optim
import torch.nn.functional as F

from dsanet.Layers import EncoderLayer, DecoderLayer
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *

class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            batch, window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob

        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        # print(d_model)
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        # print(x.shape)

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        # print(x.shape)
        # print(self.conv2(x).shape)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)

        x = torch.squeeze(x2, 2)
        # print(x.shape)
        x = torch.transpose(x, 1, 2)
        # print(x.shape)
        src_seq = self.in_linear(x)
        # print(src_seq.shape)
        enc_slf_attn_list = []

        enc_output = src_seq
        # print(enc_output.shape)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        # print(enc_slf_attn_list)
        # print(enc_output.shape)
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            batch, window, local, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(nn.Module):

    def __init__(self, window, n_multiv):

        super(AR, self).__init__()
        #self.linear = nn.Linear(window, 1) #self.linear=nn.Linear(windows, 40)
        self.linear_1=nn.Linear(window, 40)
        self.linear_2=nn.Linear(n_multiv, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.linear_2(x)
        return x


class DSANet(pl.LightningModule):
    def __init__(self, hparams):
        print("----------------------initiate model-------------------------")
        """
        Pass in parsed HyperOptArgumentParser to the model
        """
        super(DSANet, self).__init__()

        self.batch_size = hparams.batch_size

        # parameters from dataset
        self.window = hparams.window
        self.n_multiv = hparams.n_multiv
        self.local = hparams.local

        self.n_kernels = hparams.n_kernels
        self.w_kernel = hparams.w_kernel

        # hyperparameters of model
        self.d_model = hparams.d_model
        self.d_inner = hparams.d_inner
        self.n_layers = hparams.n_layers
        self.n_head = hparams.n_head
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.drop_prob = hparams.drop_prob
        self.learning_rate=hparams.lr
        self.criterion=hparams.criterion

        # build model
        self.__build_model()
        self.onetime_prediction = []
        self.label_list = []
        self.prediction = []
        self.input_for_next=0
        self.save_hyperparameters()


    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        print("----------------------build model-------------------------")
        """
        Layout model
        """
        self.sgsf = Single_Global_SelfAttn_Module(batch=self.batch_size,
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(batch=self.batch_size,
            window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR(window=self.window, n_multiv=self.n_multiv)
        #self.W_output1 = nn.Linear(2 * self.n_kernels, 1)# self.W_output1=nn.Linear(2*self.n_kernels, 40)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 40)
        self.W_output2 = nn.Linear(self.n_multiv, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.relu=nn.ReLU()
        self.batch_normalization=nn.BatchNorm1d(22)

    # ---------------------
    # TRAINING
    # ---------------------
    def loss(self, labels, predictions):

        if self.criterion == 'l1_loss':
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == 'mse_loss':

            loss = torch.sqrt(F.mse_loss(predictions, labels))
        return loss
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        """

        #print(x.shape) [batch, seq_len, feature]
        # x=self.batch_normalization(x.transpose(1,2))
        # x=torch.transpose(x, 1,2)
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        #print(sgsf_output.shape)       #[batch, feature, seq_len]
        #print(slsf_output.shape)       #[batch, feature, seq_len]
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        #print(sf_output.shape)         #[batch, feature, seq_len+seq_len]
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)
        #print(sf_output.shape)           #[batch, feature, 8]
        sf_output = torch.transpose(sf_output, 1, 2)
        #print(sf_output.shape)
        sf_output = self.W_output2(sf_output)
        #print(sf_output.shape)     #[batch, 8,1]
        ar_output = self.ar(x)
        #print(ar_output.shape)   #[batch, 8, 1]
        output = sf_output+ar_output
        #print(output.shape)

        return output

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]  # It is encouraged to try more optimizers and schedulers here
    # Define Optimizer Here

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

        MSE_loss=[]
        RMSE_loss=[]
        MAE_loss=[]
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
        write.writerow(['DSANet', MSE_average, RMSE_average, MAE_average])


