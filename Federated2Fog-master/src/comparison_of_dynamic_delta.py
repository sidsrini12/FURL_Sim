import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl


matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(20, 6))

epochs = 50

idx = 1
title = ['', '', 'a', 'b']
for line in ['accuracy', 'loss']:
    for non_iid in [1]:
        ax = fig.add_subplot(120 + idx)
        idx += 1
        files = '../history/history_mnist_svm_fog_uniform_non_iid_1_num_workers_200_lr_0.01_nest_False_batch_300_laplace_rounds_2_radius_0.7_d2d_1.0_factor_2.pkl'
        file_ = '../history/history_mnist_svm_fog_uniform_non_iid_10_num_workers_200_lr_0.01_batch_300_laplace_rounds_2_radius_0.7_d2d_1.0_factor_2_alpha_False_dyn_True_delta_0.25000000_omega_1.5_eps_mul_1.001_kappa_15.pkl'

        colors = ['r.:', 'k.:']

        x_ax, y_ax, l_test, grad_tr, _, _ = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, 'k.:', label='with delta est.')
        else:
            ax.plot(x_ax, l_test, 'k.:', label='with delta est.')

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(files, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line == 'accuracy':
            plt.plot(x_ax, y_ax, 'r.:', label='w/o delta est.')
        else:
            plt.plot(x_ax, l_test, 'r.:', label='w/o delta est.')
        ax.set_xlabel('epochs')
        ax.set_ylabel(line)
        ax.grid(True)
        ax.set_xlim(left=0, right=epochs)
        plt.title('({})'.format(title[idx]), y=-0.22)
        if idx == 3:
            ax.legend()
file_name = '../plots/_mnist_svm_fog_uniform_non_iid_10_num_workers_200_lr_0.01_batch_300_laplace_2_0.7_1.0_2_eps_mul_1.001_kappa_15'

print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.2)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
