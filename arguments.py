# coding: utf-8
import argparse
import glob
import os

#########################################################################################################
### initialize parameter ################################################################################
def parse_args(model_name = 'LSTM', results_path=os.getcwd()+'/results'):

    desc = "Tensorflow implementation of 'LSTM'"
    parser = argparse.ArgumentParser(description=desc)

    # Data parameter ##############################################################################################
    parser.add_argument('--quantilize', type=bool, default=True, help='Boolean for choosing quantilize or not')
    parser.add_argument('--n_quantilize', type=int, default=10, help='Number of quantilizations')

    # Learning parameter ##########################################################################################
    parser.add_argument('--model_name', type=str, default=model_name,
                        choices=['RNN', 'LSTM'],
                        help='ML train model name\n')
    parser.add_argument('--model_path', type=str, default=results_path+'/models', help='File path of output images')
    parser.add_argument('--results_path', type=str, default=results_path, help='File path of output images')
    parser.add_argument('--file_cnt', type=int, default=4, help='Number of data files')

    parser.add_argument('--keep_prob_cell', type=float, default=.9, help='rate dropout cell')
    parser.add_argument('--keep_prob_layer', type=float, default=.9, help='rate dropout layer')

    parser.add_argument('--n_layers', type=int, default=5, help='Number of hidden layers')

    parser.add_argument('--n_hidden', type=int, default=30, help='Number of hidden units')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--output_size', type=int, default=1, help='Output size')

    args = parser.parse_args()
    print(args)
    return check_args(args)

"""checking arguments"""
def check_args(args):

    # --results_path
    try:

        os.makedirs(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    # files = glob.glob(args.results_path+'/*')
    # for f in files:
    #     os.remove(f)

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args