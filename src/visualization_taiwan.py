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
colors_palette2_v2 = ['#ea8d5f','#80b27e']
colors_palette3 = ['#80b27e']

def corr_matrix(df):
    '''Visualizing correlation matrix'''
    
    plt.figure(figsize = (30, 20))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype = bool))

    f, ax = plt.subplots(figsize=(20, 13))
    cmap = sns.light_palette("seagreen")
    ax=sns.heatmap(corr, mask=mask, vmax=1, vmin=-1, center=0,square=True, linewidths=.5, cmap=cmap, cbar_kws={"shrink": .5}, annot=False, annot_kws={"size": 10})
    cbar=ax.collections[0].colorbar
    cbar.set_ticks([-1, -0.50, 0, 0.50, 1])

def box_plot(df):
    '''Visualizing relationship between two features'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/3), 3, figsize = (15,9))  
    y_axe = 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 3)
        sns.boxplot(x=df['class'],y=df[name], ax=axes[i,j], palette = colors_palette2)
        y_axe = y_axe + 1
    plt.tight_layout()
    plt.show()
    
def feature_plot(df,feature_name):
    '''Visualizing features'''
    fig, ax =plt.subplots(1,2,figsize=(18,8))
    sns.countplot(data = df, x = 'SEX', hue="Default", palette = 'rocket', ax=ax[0])
    ax[0].set_xlabel('Sex')
    ax[0].set_ylabel("Number of Clients", fontsize= 12)
    for p in ax[0].patches:
        ax[0].annotate((p.get_height()), (p.get_x()+0.05, p.get_height()+100))

    ax[0].set_xticks([0,1],['Male', 'Female'], fontsize = 11)
    sns.barplot(x = "SEX", y = "Default", data = df, palette = 'rocket', ci = None)
    ax[1].set_xlabel('Sex')
    ax[1].set_ylabel("% of Default", fontsize= 12)
    for p in ax[1].patches:
        ax[1].annotate("%.2f" %(p.get_height()), (p.get_x()+0.35, p.get_height()+0.003))
    ax[1].set_xticks([0,1],['Male', 'Female'], fontsize = 11)

    plt.show()
    
def distribution_plot(df):
    '''Visualizing distributions of data'''
    
    f,ax=plt.subplots(1,2,figsize=(18,8))
    df['class'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True, colors = colors_palette2)
    ax[0].set_title('target')
    ax[0].set_ylabel('')
    bar_right = sns.countplot(x=df['class'], data=df,palette=colors_palette2)
    bar_right.set_xticklabels(['Non-bankruptcy', 'Bankruptcy'])
    N, Y = df['class'].value_counts().values
    print('Number of Non-bankruptcy: ',Y)
    print('Number of Bankruptcy: ',N)
    plt.show()
    
    
def boxplot_features(df):
    ''' Box plot variables of dataset'''
    fig, axes = plt.subplots(16, 4, figsize=(25, 60))
    features = df.iloc[:,:-1].columns

    # fig.suptitle('Bankruptcy Outcome Distribution WRT All Independent Variables', fontsize=16)
    i=0
    j=0
    for elt in features:
        sns.boxplot(ax=axes[i,j], x=df['class'], y=df[elt], hue=df['class'], palette=('#23C552','#C52219'))
        axes[0, 0].set_title(elt, fontsize=12)
        j +=1
        if j == 4:
            i += 1
            j = 0

def pair_plot(df):
    '''Visualizing data with pairs plots'''
    p=sns.pairplot(df, hue = 'class', palette=colors_palette2)
    


def missing_percentage(df):
    '''Visualizing distribution of missing values'''
    percent_missing = df.isnull().mean().round(4) * 100
    trace = go.Bar(x = percent_missing.index, y = percent_missing.values ,opacity = 0.8, text = percent_missing.values.round(4),  textposition = 'auto',marker=dict(color=colors_palette3[0],
    line=dict(color='#000000',width=2)))
    layout = dict(title =  "Missing Values (count & %)")
    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)
    

def outliers_plot(df):
    '''Visualizing feature with outliers'''
    features = df.columns
    f, axes = plt.subplots(round(len(features)/4), 4, figsize = (30,60))  
    y_axe= 0
    for name in features[:-1]:
        i, j = divmod(y_axe, 4)
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
    ax1.bar(y_train_vals.index.levels[0].astype(str),y_train_vals, color = colors_palette2)
    ax2.bar(y_test_vals.index.levels[0].astype(str),y_test_vals, color = colors_palette2)
    plt.suptitle("Class distribution after splitting into train and test sets ( Count)")
    ax1.set_title("Train Set")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax2.set_title("Test Set")
    ax2.set_xlabel("Class")
    plt.show()
    
    

