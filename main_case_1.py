# coding: utf-8
import numpy as np

import arguments
import file_handler as fh
import evaluation_handler as eh

trainFile = 'head'
targetFile = 'tail'
if __name__ == '__main__':
    args = arguments.parse_args()
    s2_train_data, s2_target_data, s4_train_data, s4_target_data = fh.getData(2)

    print("Case 1")
    s_result = eh.evaluations(args, s2_train_data, s2_target_data)
    s_result = np.array(s_result)
    print('=====================================================================================================================================================')
    for i, (loss, rmse) in enumerate(s_result):
        print('fold %d: loss %03.9f rmse: %03.5f' % (i + 1, loss, rmse))
    print('=====================================================================================================================================================')
    print('Case 1 Average: loss %03.9f rmse: %03.5f' % (np.mean(s_result[:, 0]), np.mean(s_result[:, 1])))
    print('=====================================================================================================================================================')
