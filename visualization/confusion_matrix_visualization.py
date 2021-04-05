"""
TODO: write description of module
"""


import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix



def plot_and_save_confusion_matrix(y_true, y_pred, name_labels, path_to_save:str='confusion_matrix', name_filename:str='cm.png', title:str=''):
    #TODO: write description of function
    c_m = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(c_m, name_labels, name_labels)

    plt.figure(figsize=(10, 10))
    #plt.title(title, y=1., fontsize=20)

    group_counts = ['{0:0.0f}'.format(value) for value in
                    conf_matrix.values.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         conf_matrix.div(np.sum(conf_matrix, axis=1), axis=0).values.flatten()]

    labels = ['{}\n{}'.format(v1, v2) for v1, v2 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(c_m.shape)
    sns.set(font_scale=1.6)
    chart = sns.heatmap(conf_matrix,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 18},
                        cmap="Blues"
                        )
    chart.set_title(title, fontsize=20)
    chart.set_yticklabels(labels=chart.get_yticklabels(), va='center')
    chart.set_xticklabels(labels=chart.get_xticklabels(), va='center')
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save,name_filename), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

