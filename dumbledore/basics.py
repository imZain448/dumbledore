import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def vis_feature(data , feature ,target, pallete='pastel' , continous=False , target_continous=False, jitter=False):
    '''
    function to visualize and understand different features of the data set \n
    data -> dataframe \n
    feature -> feature to visualize \n
    target -> target feature you want to compare \n
    pallete -> seaborn color pallete string \n
    continous -> set this to true if you wan to visualize a continous feature \n
    target_continous -> set this true if the target is a continous variable \n
    jitter -> set true to see spread out stripplot
    '''
    sns.set_palette(pallete)

    if continous and target_continous:
        print('feature = '+feature)
        print(f'mean  = {data[feature].mean()}')
        print(f'mdeian  = {data[feature].median()}')
        print(f'std. deviation = {data[feature].std()}')
        print(f'max = {data[feature].max()}')
        print(f'min = {data[feature].min()}')
        fig = plt.figure(figsize=(14,12))
        gs = GridSpec(3,3)

        # distplot
        fig.add_subplot(gs[0,0])
        ax = sns.distplot(data[feature])
        ax.set_title('distribution of '+feature)

        #boxplot for outliers
        fig.add_subplot(gs[0,1])
        ax1 = sns.boxenplot(x=data[feature] , width=0.5 , showfliers=True)
        ax1.set_title('variability of '+feature)

        #boxplot w.r.t. target
        fig.add_subplot(gs[1,0:2])
        ax2 = sns.boxplot(x=feature, y=target , data=data , width=0.3)
        ax2.set_title('variability of '+feature+' w.r.t. '+target)

        #scatter plot w.r.t target
        fig.add_subplot(gs[:, 2])
        ax3 = sns.stripplot(x=target , y=feature, data=data , jitter=jitter)
        ax3.set_title(f'{feature} w.r.t {target}')

        #voilin plot w.r.t. target
        fig.add_subplot(gs[2,0:2])
        ax4 = sns.violinplot(x=feature , y = target , data=data)
        ax4.set_title('variability (Voilinplot) for '+ feature+ ' w.r.t. '+target)

    else:
        print(f'number of unique classes for {feature} are {data[feature].nunique()} = {data[feature].unique()}')
        print(f'value counts of each class for feature({feature}) \n {data[feature].value_counts()}')
        fig = plt.figure(figsize=(12,5))
        gs = GridSpec(1,3)
        fig.add_subplot(gs[0,0])
        ax = sns.countplot(x=feature , data=data)
        ax.set_title('Distribution of ' + feature)
        fig.autofmt_xdate()

        fig.add_subplot(gs[0,1:])
        ax2 = sns.countplot(x=target ,hue=feature, data=data)
        ax2.set_title('Distribution of '+feature+' w.r.t. churn')
        plt.legend(loc='best')
    
    fig.tight_layout(pad=3)