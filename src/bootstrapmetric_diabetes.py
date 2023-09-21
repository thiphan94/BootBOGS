import numpy as np
import time
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,roc_curve, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from datetime import datetime
# Define Evaluation Metrics
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

# Defining the model
def create_model(params):
    model = Sequential()
    model.add(Dense(params['neurons'],input_dim = 8,kernel_initializer ='normal',activation = 'relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = Adam(learning_rate = params['learning_rate'])
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

def bootstrap_metric(name_dataset, metric, accuracy_scores, auc_scores, f1_scores):
    '''
    Distributions of accuracy of AUC and of F1 score for simple bootstrap 
    '''
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Calculate mean, median and confidence intervals for each metric
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_median = np.median(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    accuracy_ci_95 = np.percentile(accuracy_scores, [2.5, 97.5])
    accuracy_ci_68 = np.percentile(accuracy_scores, [16, 84])

    auc_mean = np.mean(auc_scores)
    auc_median = np.median(auc_scores)
    auc_std = np.std(auc_scores)
    auc_ci_95 = np.percentile(auc_scores, [2.5, 97.5])
    auc_ci_68 = np.percentile(auc_scores, [16, 84])

    f1_mean = np.mean(f1_scores)
    f1_median = np.median(f1_scores)
    f1_std = np.std(f1_scores)
    f1_ci_95 = np.percentile(f1_scores, [2.5, 97.5])
    f1_ci_68 = np.percentile(f1_scores, [16, 84])


    # Set the style for the plots
    sns.set(style="whitegrid")

    # Plot the distribution of accuracy scores
    sns.kdeplot(accuracy_scores, color='blue', shade=True)
    plt.axvline(x=accuracy_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(x=accuracy_median, color='black', linestyle='dashed', linewidth=2, label='Median')
    plt.axvspan(accuracy_ci_95[0], accuracy_ci_95[1], alpha=0.3, color='blue', label='95% CI')
    plt.axvspan(accuracy_ci_68[0], accuracy_ci_68[1], alpha=0.5, color='green', label='68% CI')
    title = "Accuracy Distribution - " + name_dataset 
    plt.title(title)
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
    # plt.show()
    path_img = "../img/" + "accuracy_" + name_dataset + '_' + metric + '_' + current_datetime + ".png"
    plt.savefig(path_img)
    plt.close() # will close the plot
    print("Accuracy:")
    print(f"Mean: {accuracy_mean:.4f}")
    print(f"Standard Deviation: {accuracy_std:.4f}")
    print(f"95% Percentile Bootstrap Confidence Interval: [{accuracy_ci_95[0]:.4f}, {accuracy_ci_95[1]:.4f}]")
    print(f"68% Percentile Bootstrap Confidence Interval: [{accuracy_ci_68[0]:.4f}, {accuracy_ci_68[1]:.4f}]")
    print()


    # Plot the distribution of AUC scores
    sns.kdeplot(auc_scores, color='blue', shade=True)
    plt.axvline(x=auc_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(x=auc_median, color='black', linestyle='dashed', linewidth=2, label='Median')
    plt.axvspan(auc_ci_95[0], auc_ci_95[1], alpha=0.3, color='blue', label='95% CI')
    plt.axvspan(auc_ci_68[0], auc_ci_68[1], alpha=0.5, color='green', label='68% CI')
    title = "AUC Distribution - " + name_dataset 
    plt.title(title)
    plt.xlabel('AUC')
    plt.ylabel('Density')
    plt.legend()
    # plt.show()
    path_img = "../img/" + "auc_" + name_dataset + '_' + metric + '_' + current_datetime + ".png"
    plt.savefig(path_img)
    plt.close() # will close the plot
    print("AUC:")
    print(f"Mean: {auc_mean:.4f}")
    print(f"Standard Deviation: {auc_std:.4f}")
    print(f"95% Percentile Bootstrap Confidence Interval: [{auc_ci_95[0]:.4f}, {auc_ci_95[1]:.4f}]")
    print(f"68% Percentile Bootstrap Confidence Interval: [{auc_ci_68[0]:.4f}, {auc_ci_68[1]:.4f}]")
    print()

    # Plot the distribution of F1 scores
    sns.kdeplot(f1_scores, color='blue', shade=True)
    plt.axvline(x=f1_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(x=f1_median, color='black', linestyle='dashed', linewidth=2, label='Median')
    plt.axvspan(f1_ci_95[0], f1_ci_95[1], alpha=0.3, color='blue', label='95% CI')
    plt.axvspan(f1_ci_68[0], f1_ci_68[1], alpha=0.5, color='green', label='68% CI')
    title = "F1 Distribution - " + name_dataset 
    plt.title(title)
    plt.xlabel('F1 Score')
    plt.ylabel('Density')
    plt.legend()
    # plt.show()
    path_img = "../img/" + "f1_" + name_dataset + '_' + metric + '_' + current_datetime + ".png"
    plt.savefig(path_img)
    plt.close() # will close the plot
    print("F1 Score:")
    print(f"Mean: {f1_mean:.4f}")
    print(f"Standard Deviation: {f1_std:.4f}")
    print(f"95% Percentile Bootstrap Confidence Interval: [{f1_ci_95[0]:.4f}, {f1_ci_95[1]:.4f}]")
    print(f"68% Percentile Bootstrap Confidence Interval: [{f1_ci_68[0]:.4f}, {f1_ci_68[1]:.4f}]")

