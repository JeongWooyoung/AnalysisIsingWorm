# coding: utf-8
import numpy as np
from datetime import datetime

import arguments
import file_handler as fh
import evaluation_handler as eh

trainFile = 'head'
targetFile = 'tail'
if __name__ == '__main__':
    start = datetime.now()
    # temperature = ['1.000', '1.500', '2.000', '2.500', '3.000', '3.500', '4.000', '4.500', '5.000']
    temperature = ['2.500']
    print('=====================================================================================================================================================')
    args = arguments.parse_args()
    args.file_cnt=1
    train_input, train_target, test_input, test_target = fh.getWormData(args.file_cnt, temperature)
    print('=====================================================================================================================================================')

    layers = 5
    nodes = 10
    for layer in range(0, layers):
        for node in range(0, nodes):
            args.n_layers = layer+1
            args.n_hidden = node+1

            print("======================================================= Case 3. %02d layers, %02d hidden nodes =========================================================="
                  %(args.n_layers, args.n_hidden))
            s_result = eh.evaluations3(args, train_input, train_target, test_input, test_target)
            s_result = np.array(s_result)
            print('=====================================================================================================================================================')
            for i, (loss, rmse) in enumerate(s_result):
                print('fold %d: loss %03.9f rmse: %03.5f' % (i + 1, loss, rmse))
            print('=====================================================================================================================================================')
            print('Case 3 Average: loss %03.9f rmse: %03.5f' % (np.mean(s_result[:, 0]), np.mean(s_result[:, 1])))
            print('=====================================================================================================================================================')

    # print("======================================================= Case 3. =====================================================================================")
    # s_result = eh.evaluations2(args, train_input, train_target, test_input, test_target)
    # s_result = np.array(s_result)
    # print('=====================================================================================================================================================')
    # for i, (loss, rmse) in enumerate(s_result):
    #     print('fold %d: loss %03.9f rmse: %03.5f' % (i + 1, loss, rmse))
    # print('=====================================================================================================================================================')
    # print('Case 3 Average: loss %03.9f rmse: %03.5f' % (np.mean(s_result[:, 0]), np.mean(s_result[:, 1])))
    # print('=====================================================================================================================================================')
    # print(datetime.now()-start)
