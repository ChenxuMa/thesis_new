import argparse
from data_globalNorm import Data
import os
from run import run
import csv







if __name__=="__main__":
    # predict_file_name="./prediction_DSANet(openDD).csv"
    # figure_folder_name="./figure_DSANet(openDD)"
    # result_file_name="result_DSANet(openDD).csv"
    result_file_name = "result.csv"
    data_file_path="./dataset/vehiclesData"
    data_evluation_file = "./MATLAB_dataset/dataEvaluation.csv"
    normalization_parameter_file = "nomalization_params.csv"
    root_dir = os.path.dirname(os.path.realpath(__file__))
    print(root_dir)
    demo_log_dir = os.path.join(root_dir, 'dsanet_logs')
    print(demo_log_dir)
    checkpoint_dir = os.path.join(demo_log_dir, "model_weights")
    print(checkpoint_dir)



    parent_parser=argparse.ArgumentParser(description='Process argument')
    parent_parser.add_argument('--model', type=str, default=None,help='must specify which model to train')
    parent_parser.add_argument('--data_name', type=str)
    parent_parser.add_argument('--dataset_path', default="./dataset/vehiclesData", type=str)
    parent_parser.add_argument('--use_GPU', default=True, type=bool)
    parent_parser.add_argument('--n_multiv', default=22, type=int)
    parent_parser.add_argument('--window', default=10, type=int)
    parent_parser.add_argument('--horizon', default=1, type=int)
    parent_parser.add_argument('--output_length', default=40, type=int)

    # training params (opt)
    parent_parser.add_argument('--lr', default=0.005, type=float)
    parent_parser.add_argument('--optimizer_name', default='adam', type=str)
    parent_parser.add_argument('--criterion', default='mse_loss', type=str)


    parent_parser.add_argument('--batch_size', default=32, type=int,
                           help='batch size will be divided over all the gpus being used across all nodes')
    parent_parser.add_argument('--max_epochs', default=50, type=int)
    # DSANet Params
    parent_parser.add_argument('--local', default=3, type=int)
    parent_parser.add_argument('--n_kernels', default=22, type=int)
    parent_parser.add_argument('--w_kernel', type=int, default=1)
    parent_parser.add_argument('--d_model', type=int, default=512)
    parent_parser.add_argument('--d_inner', type=int, default=1024)
    parent_parser.add_argument('--d_k', type=int, default=64)
    parent_parser.add_argument('--d_v', type=int, default=64)
    parent_parser.add_argument('--n_head', type=int, default=8)
    parent_parser.add_argument('--n_layers', type=int, default=5)
    parent_parser.add_argument('--drop_prob', type=float, default=0.2)

    #LSTM params
    parent_parser.add_argument('--n_out_multiv', default=1, type=int)
    parent_parser.add_argument('--n_hidden', default=512, type=int)



    hyperparams = parent_parser.parse_args()

    # run on HPC cluster

    # * change the following code to comments for grid search


    data_initialization = Data(hyperparams, data_file_path)
    data_initialization.prepare_data()
    train_loader = data_initialization.train_dataloader()
    val_loader = data_initialization.val_dataloader()

    # scaler_dict = data_initialization.get_scaler_dict()
    scaler_dict=data_initialization.get_scaler_dict()
    mean_dict=data_initialization.get_mean_dict()
    std_dict=data_initialization.get_std_dict()

    header=['feature', 'mean', 'std']
    file = open(normalization_parameter_file, 'a', newline='')
    write = csv.writer(file)
    write.writerow(header)
    for k, v in mean_dict.items():
        write = csv.writer(file)
        write.writerow([k, mean_dict[k], std_dict[k]])

    test_loader = data_initialization.test_dataloader()
    models_result=run(hyperparams,checkpoint_dir).train(train_loader, val_loader)

    models_result.test_model(test_loader,result_file_name)




