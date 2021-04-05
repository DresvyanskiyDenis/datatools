"""
TODO: write description of module
"""



import os
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score



def plot_weights(weights, path_to_save:str='confusion_matrix', name_filename:str='cm.png', title:str=''):
    # TODO: write description of function
    weights=weights.T
    weights=pd.DataFrame(data=weights, index=['AffectNet','AffWild2','Audio M.', 'L-SVM'],
                         columns=['Neutral','Anger','Disgust','Fear',
                                                    'Happiness','Sadness','Surprised'])
    #plt.figure(figsize=(10, 10))
    group_percentages = ['{0:.2f}'.format(value) for value in
                         weights.values.flatten()]
    labels = ['{}'.format(v1) for v1 in group_percentages]
    labels = np.asarray(labels).reshape(weights.shape)
    sns.set(font_scale=1.3)
    chart = sns.heatmap(weights,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 15},
                        cmap="Blues"
                        )
    chart.set_yticklabels(labels=chart.get_yticklabels(), va='center')
    chart.set_xticklabels(labels=chart.get_xticklabels(), va='top', ha='center', rotation=45)
    #chart.set_title(title, fontsize=14)
    #plt.ylabel("Class weights")
    #plt.xlabel("Models")
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save, name_filename), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def plot_f1_scores_of_models_from_predictions(predictions:List[np.ndarray], ground_truth:np.array, path_to_save:str='confusion_matrix',
                                                   name_filename:str='cm.png', title:str='', num_classes:int=7):
    # TODO: write description of function
    ground_truth=ground_truth.reshape((-1,1))

    f1_scores=[]
    for pred_idx in range(len(predictions)):
        curr_predictions=predictions[pred_idx].reshape((-1,1))
        f1=[]
        for class_idx in range(num_classes):
            f1.append(f1_score(ground_truth, curr_predictions, average='macro', labels=[class_idx]))
        f1_scores.append(f1)



    f1_scores=np.array(f1_scores)
    labels=np.array(['{0:.2f}'.format(value) for value in f1_scores.reshape((-1,))]).reshape(f1_scores.shape)
    f1_scores=pd.DataFrame(f1_scores, columns=['Neutral','Anger','Disgust','Fear',
                                                    'Happiness','Sadness','Surprised'],
                         index=['AffectNet','AffWild2','Audio M.', 'L-SVM', 'Fusion'])

    sns.set(font_scale=1.3)
    chart = sns.heatmap(f1_scores,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 15},
                        cmap="Blues"
                        )
    chart.set_yticklabels(labels=chart.get_yticklabels(), va='center')
    chart.set_xticklabels(labels=chart.get_xticklabels(),va='center')
    #chart.set_title(title, fontsize=16)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save, name_filename), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

