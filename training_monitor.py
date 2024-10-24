import os

from tensorflow.keras.callbacks import Callback

import numpy as np
import matplotlib.pyplot as plt

def plot_fea_importance_hist(fea_importance=None, name=None, tick_names=None, method_name='', show_order=True, dataset=None, headings=None):
    """
    Visualize the feature importance scores (e.g. feature masks).
    :param fea_importance: The feature importance vector or feature mask.
    :param name: The file name for saving the plot.
    :param tick_names: The names of the input variables. It can be automatically generated if not specified.
    :param method_name: The name of the variable selection method.
    :param show_order: Whether to show the order of the input variables in the plot.
    :return: None.
    """

    fea_order = (np.argsort(fea_importance)[::-1]).argsort() + 1
    # 对给定的特征重要度进行排序，并将排序后的索引（从1开始）作为新的排序
    if tick_names is not None:
        ordered_variables = [tick_names[i] for i in (np.argsort(fea_importance)[::-1])]
        print(ordered_variables)

    if len(fea_importance) > 10:
        plt.figure(figsize=(16, 4.5), dpi=200)
    else:
        plt.figure(figsize=(10, 4.), dpi=200)

    # plot feature importance scores using bar plots
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=.55)
    bars = plt.bar(np.arange(0, len(fea_importance)), fea_importance, alpha=0.65)

    # annotate feature orders on the top of each bar
    # if show_order:
    #     for idx, rect in enumerate(bars):
    #         plt.annotate(text=r'(%d)' % fea_order[idx],
    #                      xy=[rect.get_x() + rect.get_width() / 2, rect.get_height()],
    #                      fontsize=16, c='black', fontweight='bold', ha='center', va='bottom')
    if show_order:
        for idx, rect in enumerate(bars):
            if dataset in ['Random_Dataset']:
                plt.annotate(text=r'(%d)' % fea_order[idx],
                             xy=[rect.get_x() + rect.get_width() / 2, rect.get_height()],
                             fontsize=16, c='black', fontweight='bold', ha='center', va='bottom')

            if dataset in ['MNIST']:
                if fea_order[idx] < 16:
                    plt.annotate(text=r'(%d)' % fea_order[idx],
                                 xy=[rect.get_x() + rect.get_width() / 2, rect.get_height()],
                                 fontsize=8, c='black', fontweight='bold', ha='center', va='bottom')

            elif dataset in ['NAKO_Diabetes_Dataset', 'NAKO_Normal_Dataset']:
                if fea_order[idx] < 16:
                    plt.annotate(text=r'(%d: %s)' % (fea_order[idx], headings[idx]),
                                 xy=[rect.get_x() + rect.get_width() / 2, rect.get_height()],
                                 fontsize=8, c='black', fontweight='bold', ha='center', va='bottom')

    if dataset in ['Random_Dataset', 'NAKO_Diabetes_Dataset', 'NAKO_Normal_Dataset']:
        step = 10
    elif dataset in ['MNIST']:
        step = 100

    if tick_names is not None:
        reduced_ticks = np.arange(0, len(tick_names), step)
        reduced_labels = [tick_names[i] for i in reduced_ticks]
        plt.xticks(fontsize=20, ticks=np.arange(0, len(tick_names), step), labels=reduced_labels)
    elif len(fea_importance) <= 32:
        plt.xticks(fontsize=20, ticks=[n for n in range(len(fea_importance))],
                   labels=[r'$x_{%d}$' % (n+1) for n in range(len(fea_importance))])
    else:
        plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)
    plt.xlabel(r'Input variables', fontsize=22)
    plt.ylabel(r'Importance scores', fontsize=22)
    plt.ylim([0, np.max(fea_importance)*1.2])
    plt.title(label=r'%s' % method_name, loc='right', fontsize=22)
    folder_path = "Experiments"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, 'FS_%s.png' % name)
    plt.savefig(file_path, format='png', dpi=200, bbox_inches='tight')
    plt.close()

class SelectorMonitor(Callback):
    def __init__(self, x=None, y=None, method=None, save_name=None, step=10, tick_names=None, top_k=None, dataset=None):
        """
        A callback object for monitoring the training procedure.
        :param x: The input dataset.
        :param y: The target variables.
        :param method: The name of the variable selection method.
        :param save_name: The file name for saving the plot.
        :param step: The period of monitoring the training procedure.
        :param tick_names: The names of the input variables. It can be automatically generated if not specified.
        :param top_k: The desired number of features to be selected.
        """
        super(SelectorMonitor, self).__init__()
        self.x = x
        self.y = y
        self.method = method
        self.save_name = save_name
        self.step = step
        self.tick_names = tick_names
        self.top_k = top_k
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            m = self.model.get_layer(self.method).get_support(self.x)
            print('Epoch: %d -- Loss: %.4f' % (epoch, logs['loss']))
            plot_fea_importance_hist(fea_importance=m, name='%s_%s_ep_%d' % (self.save_name, self.dataset, epoch),
                                     tick_names=self.tick_names, method_name=self.method, show_order=False, dataset=self.dataset)
            # print(m)

