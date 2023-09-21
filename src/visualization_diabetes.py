# Plots
import seaborn as sns
sns.set()
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
# %matplotlib inline
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import missingno as msno
from sklearn.metrics import auc, confusion_matrix,accuracy_score,roc_curve, classification_report, f1_score, roc_auc_score,log_loss,recall_score, precision_score, f1_score,precision_recall_curve
import numpy as np

colors_palette1 = ['#80b27e', '#abc78d', '#d6dca0', '#fff1b9', '#f7d193', '#f1b075', '#ea8d5f']
colors_palette2 = ['#80b27e','#ea8d5f']
colors_palette3 = ['#80b27e']

def corr_matrix(df):
    '''Visualizing correlation matrix'''
    plt.figure(figsize=(8,6)) 
    p=sns.heatmap(df.iloc[:,:-1].corr(), annot=True,cmap = colors_palette1) 

def box_plot(df):
    '''Visualizing relationship between two features'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/3), 3, figsize = (15,9))  
    y_axe = 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 3)
        sns.boxplot(x=df['Outcome'],y=df[name], ax=axes[i,j], palette = colors_palette2)
        y_axe = y_axe + 1
    plt.tight_layout()
    plt.show()
    
def feature_plot(df):
    '''Visualizing features'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/3), 3, figsize = (15,9))  
    y_axe = 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 3)
        sns.boxplot(x=df[name],data=df, ax=axes[i,j], palette = colors_palette2)
        y_axe = y_axe + 1
    plt.tight_layout()
    plt.show()
    
def distribution_plot(df):
    '''Visualizing distributions of data'''
    f,ax=plt.subplots(1,2,figsize=(18,8))
    df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True, colors = colors_palette2)
    ax[0].set_title('target')
    ax[0].set_ylabel('')
    bar_right = sns.countplot(x=df['Outcome'], data=df,palette=colors_palette2)
    bar_right.set_xticklabels(["Healthy","Diabetic"])
    healthy, diabetics = df['Outcome'].value_counts().values
    print("Samples of diabetic people: ", diabetics)
    print("Samples of healthy people: ", healthy)
    plt.show()
    
def distribution_features(df):
    '''Visualizing distributions of feature against target'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/3), 3, figsize = (15,10))  
    y_axe = 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 3)
        sns.histplot(df[name][df.Outcome == 1], color =colors_palette2[0], ax=axes[i, j], kde=True)
        sns.histplot(df[name][df.Outcome == 0], color =colors_palette2[1], ax=axes[i, j], kde=True)
        # sns.distplot(df[name][df.Outcome == 1], color =colors_palette2[0], rug = True, ax=axes[i, j])
        # sns.distplot(df[name][df.Outcome == 0], color =colors_palette2[1],rug = True, ax=axes[i, j])
        # plt.legend(['Diabetes', 'No Diabetes'])
        y_axe = y_axe + 1
    f.legend(['Diabetes', 'No Diabetes'])
    plt.tight_layout()
    plt.show()
    
def pair_plot(df):
    '''Visualizing data with pairs plots'''
    p=sns.pairplot(df, hue = 'Outcome', palette=colors_palette2)
    
    
def violin_plot(df):
    '''Visualizing data with violin plots'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/3), 3, figsize = (15,10))  
    y_axe = 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 3)
        sns.violinplot(data=df, x=name,color = colors_palette3[0], ax =axes[i, j])
        plt.legend(['Diabetes', 'No Diabetes'])
        y_axe= y_axe + 1
    plt.tight_layout()
    plt.show()
    
def missing_plot(df):
    '''Visualizing data with missing values'''
    p = msno.bar(df, figsize=(10,5), fontsize=12, color  = colors_palette3)

    
def missing_percentage(df):
    '''Visualizing distribution of missing values'''
    percent_missing = df.isnull().mean().round(4) * 100
    trace = go.Bar(x = percent_missing.index, y = percent_missing.values ,opacity = 0.8, text = percent_missing.values.round(4),  textposition = 'auto',marker=dict(color=colors_palette3[0],
    line=dict(color='#000000',width=1.25)))
    layout = dict(title =  "Missing Values (count & %)")
    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)
    

def outliers_plot(df):
    '''Visualizing feature with outliers'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/3), 3, figsize = (15,10))  
    y_axe= 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 3)
        sns.boxplot(x=df[name], ax=axes[i, j], color =colors_palette3[0])
        y_axe = y_axe + 1
    plt.tight_layout()
    plt.show()
    
    
def split_plot(y_train, y_test):
    '''Visualizing dataset before and after splitting dataset'''
    # Get the value counts of our train and test sets
    y_train_vals = y_train.value_counts()
    y_test_vals = y_test.value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax1.bar(y_train_vals.index.astype(str),y_train_vals, color = colors_palette2)
    ax2.bar(y_test_vals.index.astype(str),y_test_vals, color = colors_palette2)
    plt.suptitle("Class distribution after splitting into train and test sets ( Count)")
    ax1.set_title("Train Set")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax2.set_title("Test Set")
    ax2.set_xlabel("Class")
    plt.show()
    
    

