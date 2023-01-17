import os
import random
from random import shuffle
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pre_process_data





class data_process:

    def __init__(self, dataset_path, sequence_length, output_length,
                 data_file_path,
                 train_test_percentage,
                 validation_percentage):


        self.dataset_path=dataset_path
        self.data_file_path=data_file_path
        self.output_length=output_length
        self.sequence_length=sequence_length
        #self.folder_list=self.reorder_file(self.dataset_path)


        self.feature_list = ['vehicleAcceleration', 'dist2inter',
                             'distanceToVehicle', 'velocityLeading',
                             'p1', 'n1', 'p2', 'n2', 'p3', 'n3', 'p4', 'n4', 'p5', 'n5', 'p6', 'n6',
                             'interDist1', 'interDist2',
                             'interVel1', 'interVel2',
                             'interPriority1', 'interPriority2'
                             ]
        self.normalized_df, self.file_order, self.table_information, self.mean_dict, self.std_dict, self.scaler_dict\
            = self.generate_normalization_params(self.data_file_path)
        self.total_inout=self.generate_sequence_and_label(self.normalized_df, self.file_order, self.table_information, self.data_file_path)
        self.random_data=self.shuffle_data(self.total_inout)
        self.train_test_percentage = train_test_percentage
        self.validation_percentage = validation_percentage
        self.train_list, self.val_list, self.test_list= self.train_test_split(self.random_data,
                                                                            self.train_test_percentage,
                                                                            self.validation_percentage)
    def generate_sequence_and_label(self, normalized_df, file_order, table_information, data_file_path):
        total_inout=[]
        for item in tqdm(file_order):
            for car in item[1]:
                file_name=str(item[0])+"_"+str(car)+".csv"
                directory = data_file_path + "/" + file_name
                left_bound = 0
                right_bound = self.sequence_length
                while right_bound<=table_information[file_name] \
                        and left_bound<=table_information[file_name] \
                        and right_bound-left_bound==self.sequence_length:
                    if right_bound - left_bound == self.sequence_length:
                        sequence = torch.tensor(
                            normalized_df[left_bound:right_bound][self.feature_list].astype('float32').values)
                        label = torch.tensor(
                            normalized_df.iloc[right_bound - 1]['Acceleration label 1':'Acceleration label 40'].astype(
                                'float32').values).unsqueeze(1)

                        total_inout.append((sequence, label))
                    right_bound = right_bound + 1
                    left_bound = left_bound + 1



        return total_inout
    def generate_normalization_params(self, data_file_path):
        # file_list = self.reorder_data_file(data_file_path)
        file_order={}

        table_information={}
        for file in tqdm(os.listdir(data_file_path)):

            number=int(file.split("_")[0])
            car=int(file.split("_")[1][:-4])
            dp = pd.read_csv(data_file_path + "/" + file)
            table_information[file]=dp.shape[0]
            if number not in file_order.keys():
                file_order[number]=[]
                file_order[number].append(car)
            else:
                file_order[number].append(car)

            file_order[number]=sorted(file_order[number], reverse=False)
        file_order=sorted(file_order.items(), key=lambda x:x[0], reverse=False)
        total_files = []
        mean_dict = {}
        std_dict={}
        for item in tqdm(file_order):
            for car in item[1]:
                dp = pd.read_csv(data_file_path + "/" + str(item[0])+"_"+str(car)+".csv")
                total_files.append(dp)
        total_data_frame = pd.concat(total_files, axis=0, ignore_index=True)

        StandardScaler_dict = {}
        normalized_dict = {}
        for feature in self.feature_list:
            if feature not in StandardScaler_dict.keys():
                StandardScaler_dict[feature] = StandardScaler()
                total_data_frame[feature] = StandardScaler_dict[feature] \
                    .fit_transform(total_data_frame[feature].values.reshape(-1, 1)
                                   .astype('float32'))

        normalized_df=total_data_frame

        for feature in self.feature_list:
            mean_dict[feature]=StandardScaler_dict[feature].mean_[0]
            std_dict[feature]=StandardScaler_dict[feature].scale_[0]
        return normalized_df, file_order, table_information, mean_dict, std_dict, StandardScaler_dict

    def train_test_split(self, random_data, train_test_percentage, validation_percentage):

        # len_of_test_data=int(len(random_order_data.get(car))*(1-train_test_percentage))

        len_of_validation_and_train=int(len(random_data) * 0.8)
        #len_of_validation_and_train = len(random_car_and_their_data.get(car)) - 1
        len_of_validation_data = int(len_of_validation_and_train * validation_percentage)
        len_of_train_data = len_of_validation_and_train - len_of_validation_data

        train_list=random_data[0:len_of_train_data]
        val_list=random_data[len_of_train_data: len_of_train_data+len_of_validation_data]
        test_list=random_data[len_of_validation_and_train:]

        return train_list, val_list, test_list

    def shuffle_data(self, total_inout):


        random.shuffle(total_inout)

        return total_inout






