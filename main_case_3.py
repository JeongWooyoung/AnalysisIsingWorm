# coding: utf-8
import numpy as np

import arguments
import file_handler as fh
import evaluation_handler as eh

trainFile = 'head'
targetFile = 'tail'
if __name__ == '__main__':
    file_cnt=2
    args = arguments.parse_args()
    s2_train_input, s4_train_target, s2_test_input, s4_test_target = fh.getWormData(file_cnt)

    print("Case 3")
    s_result = eh.evaluations2(args, s2_train_input, s4_train_target, s2_test_input, s4_test_target)
    s_result = np.array(s_result)
    print('=====================================================================================================================================================')
    for i, (loss, rmse) in enumerate(s_result):
        print('fold %d: loss %03.9f rmse: %03.5f' % (i + 1, loss, rmse))
    print('=====================================================================================================================================================')
    print('Case 3 Average: loss %03.9f rmse: %03.5f' % (np.mean(s_result[:, 0]), np.mean(s_result[:, 1])))
    print('=====================================================================================================================================================')
