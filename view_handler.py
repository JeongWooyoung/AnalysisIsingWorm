# coding: utf-8

#########################################################################################################
## Viewer ###############################################################################################
import file_handler as fh
import matplotlib.pyplot as plt
import arguments

def showScatter(Xs, Ys, check_size=None, args=None):
    cnt = len(Xs)
    colors = ['green', 'blue', 'yellow', 'red', 'black']
    if check_size is None:
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            plt.scatter(x[:check_size], y[:check_size], color=colors[i])
    else:
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            plt.scatter(x, y, color=colors[i])
    plt.show()
def saveScatter(Xs, Ys, check_size=None, args=None):
    cnt = len(Xs)
    colors = ['green', 'blue', 'yellow', 'red', 'black']
    if check_size is None:
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            plt.scatter(x[:check_size], y[:check_size], color=colors[i])
    else:
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            plt.scatter(x, y, color=colors[i])
    if args is None:
        args = arguments.parse_args()
    fig_path = fh.getStoragePath()+'Figures/Files_%d/Layer_%d_Hidden_%d/'%(args.file_cnt, args.n_layers, args.n_hidden)
    fh.makeDirectories(fig_path)
    plt.savefig(fig_path+'Epoch_%d.png'%(args.num_epochs))
    plt.close('all')