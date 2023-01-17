import csv

from tqdm import tqdm
import os
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# def rename(dataset_path):
#     for file_name in os.listdir(dataset_path):
#         if "dataFitered.csv" in file_name:
#             index_array = file_name.split("_")
#             index = index_array[0]
#             sub_index = index_array[1]
#             if len(sub_index) == 1:
#                 sub_index = "0" + sub_index
#             os.rename(dataset_path + "/" + file_name, dataset_path + "/" + index + "." + sub_index + ".csv")


def rewrite_data(dataset_path, new_data_path):
    # file_list = reorder_file(dataset_path)
    car_list, total_data_frame, file_position= generate_car_list(file_list)
    for car in tqdm(car_list):
        file_first_line = 0
        folder_path=os.path.join(new_data_path, str(car))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            pass
        for k, v in file_position.items():
            file_path=os.path.join(folder_path, str(k))
            file_end_line = file_first_line + v - 1
            file_df_car = total_data_frame[file_first_line:file_end_line].loc[
                (total_data_frame.vehicleId == car)].reset_index(drop=True)

            file_first_line = file_end_line + 1
            #print(file_df_car)
            if file_df_car.shape[0]>=sequence_length:
                file_df_car.to_csv(file_path, index=False)


def generate_car_list(file_list):
    total_files = []
    file_position = {}
    for item in tqdm(file_list):
        dp = pd.read_csv(dataset_path + "/" + item)
        file_position[item] = dp.shape[0]
        total_files.append(dp)
    total_data_frame = pd.concat(total_files, axis=0, ignore_index=True)
    car_list = total_data_frame['vehicleId'].unique()
    normalized_input_dict, scaler_dict= normalize_input_and_output(total_data_frame,feature_list)
    for item in feature_list:
        total_data_frame[item] = normalized_input_dict[item]
    # for key, value in normalized_label_dict.items():
    #     total_data_frame[key] = normalized_label_dict[key]
    return car_list, total_data_frame, file_position

def normalize_input_and_output(data_sequence, feature_list):
    StandardScaler_dict = {}
    normalized_dict = {}
    label_scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_label_dict = {}
    #print(data_sequence['vehicleVelocity'])


    for feature in feature_list:
        if feature not in StandardScaler_dict.keys():
            StandardScaler_dict[feature] = StandardScaler()
            normalized_dict[feature] = StandardScaler_dict[feature] \
                .fit_transform(data_sequence[feature].values.reshape(-1, 1)
                               .astype('float32'))


    # for label in label_list:
    #     if label not in StandardScaler_dict.keys():
    #         StandardScaler_dict[label] = StandardScaler()
    #         normalized_label_dict[label] = StandardScaler_dict[label].fit_transform(
    #             data_sequence[label].values.reshape(-1, 1)
    #             .astype('float32'))
            # print("origin")
            # print(data_sequence[label][:10])
            # print(StandardScaler_dict[label].inverse_transform(normalized_label_dict[label][:10]))
            # print(label)
            # print(StandardScaler_dict[label].mean_)

    #print(normalized_dict['vehicleVelocity'][:10, :])
    return normalized_dict, StandardScaler_dict

def reorder_file(dataset_path):
    file_list = os.listdir(dataset_path)
    # if dataset_path == "./dataset":
    #     file_list.remove("DatasetLearningProblem")
    file_list.sort(key=lambda x: str(x[:-4]))
    return file_list


if __name__ == "__main__":

    parent_parser=argparse.ArgumentParser(description='Process argument')
    parent_parser.add_argument('--dataset_path', default="./dataset/vehiclesData", type=str)
    parent_parser.add_argument('--new_data_path', default="./dataset/vehiclesData_normalized", type=str)
    parent_parser.add_argument('--sequence_length', default=40, type=int)
    hyperparams = parent_parser.parse_args()
    new_data_path=hyperparams.new_data_path

    dataset_path=hyperparams.dataset_path
    sequence_length=hyperparams.sequence_length


    rename(dataset_path)
    rewrite_data(dataset_path, new_data_path)