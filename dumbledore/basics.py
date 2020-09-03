import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as  sns
from scipy import stats
from matplotlib.gridspec import GridSpec

def vis_feature(data , feature ,target, pallete='gist_earth' , continous=False ,target_continous=False, jitter=False , save_fig=False):
    '''
    function to visualize and understand different features of the data set \n
        data(pd.dataframe) :  dataframe 
        feature(str) : feature to visualize 
        target(str) : target feature you want to compare 
        pallete(str) : seaborn color pallete string 
        continous(bool) : set this to true if you wan to visualize a continous feature 
        target_continous(bool) : set this to true if your target variable is continous 
        jitter(float/bool) : ammount of jitter for stripplot it is only required when you 
        \t\tare visualizing a continous varibale with respect to categorical target
        save_fig(bool) : if True saves the fig as plot.png in the working directory

    returns
        pandas.DataFrame object : dataframe containing the stats for the feature
    '''
    sns.set_palette(pallete)
    da = None
    # varibale cont , target cat
    if continous and not target_continous:
        # update
        n_classes = data[target].unique()
        print('feature = '+feature)
        print(f'mean  = {data[feature].mean()}')
        print(f'mdeian  = {data[feature].median()}')
        print(f'std. deviation = {data[feature].std()}')
        print(f'max = {data[feature].max()}')
        print(f'min = {data[feature].min()}')
        print('Distribtuion accross the target : ' +target)
        df = data[feature]
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3 - q1
        lb = q1 - 1.5*IQR
        ub = q1 + 1.5*IQR
        out_low = len(df[df < lb])
        out_high = len(df[df > ub])
        print(f'IQR = {IQR} ')
        print(f'total outliers = {out_low + out_high}')
        print(f'number outliers on the right = {out_high} , (upper_bound = {ub})')
        print(f'number outliers on the left = {out_low} , (lower_bound = {lb})')
        df = data.loc[data[target] == n_classes[0], feature]
        emp = {}
        emp.update({'target' : n_classes[0]})
        emp.update({'mean':df.mean()})
        emp.update({'median':df.median()})
        emp.update({'std':df.std()})
        emp.update({'max' :df.max()})
        emp.update({'min':df.min()})
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3 - q1
        lb = q1 - 1.5*IQR
        ub = q1 + 1.5*IQR
        out_low = len(df[df < lb])
        out_high = len(df[df > ub])
        emp.update({'IQR' :IQR})
        emp.update({'total outliers' :out_low + out_high})
        emp.update({'number higher outliers' :out_high })
        emp.update({'upper bound': ub})
        emp.update({'number lower outliers':out_low})
        emp.update({'lower bound':lb})
        da = pd.DataFrame(columns=emp.keys())
        da = da.append(emp , ignore_index=True)
        for c in n_classes[1:]:
            df = data.loc[data[target] == c, feature]
            emp = {}
            emp.update({'target' : c})
            emp.update({'mean':df.mean()})
            emp.update({'median':df.median()})
            emp.update({'std':df.std()})
            emp.update({'max' :df.max()})
            emp.update({'min':df.min()})
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            IQR = q3 - q1
            lb = q1 - 1.5*IQR
            ub = q1 + 1.5*IQR
            out_low = len(df[df < lb])
            out_high = len(df[df > ub])
            emp.update({'IQR' :IQR})
            emp.update({'total outliers' :out_low + out_high})
            emp.update({'number higher outliers' :out_high })
            emp.update({'upper bound': ub})
            emp.update({'number lower outliers':out_low})
            emp.update({'lower bound':lb})
            da = da.append(emp , ignore_index=True)
        da = da.set_index('target')
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

    #varibale cont target cont
    elif continous and target_continous:
        print('feature = '+feature)
        emp = {}
        emp.update({'mean':data[feature].mean()})
        emp.update({'mdeian':data[feature].median()})
        emp.update({'std.':data[feature].std()})
        emp.update({'max':data[feature].max()})
        emp.update({'min':data[feature].min()})
        df = data[feature]
        dt = data[target]
        c_ov = np.cov(df, dt)
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3 - q1
        lb = q1 - 1.5*IQR
        ub = q1 + 1.5*IQR
        out_low = len(df[df < lb])
        out_high = len(df[df > ub])
        emp.update({'IQR':IQR})
        emp.update({'total outliers':out_low + out_high})
        emp.update({'number outliers on the right':out_high})
        emp.update({'upper_bound':ub})
        emp.update({'number outliers on the left':out_low})
        emp.update({'lower_bound':lb})
        emp.update({f"covariance of {feature} with {target}":c_ov[0,1]})
        pearson, _ = stats.pearsonr(df, dt)
        emp.update({"Pearnson's Correlation":pearson})
        spearman, _ = stats.spearmanr(df, dt)
        emp.update({"Spearman's Correlation":spearman})
        
        da = pd.DataFrame(emp.values() ,columns=[feature] ,index=emp.keys())
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3,4)

        #distplot
        fig.add_subplot(gs[0,0])
        ax1 = sns.kdeplot(data[feature] , shade=True)
        ax1.set_title('Kernel Density of '+feature)

        ax2 = fig.add_subplot(gs[1,0])
        ax = plt.scatter(x=data[feature] ,y=data[target])
        ax2.set_title('relationship with '+target+ '\nScatterplot')

        fig.add_subplot(gs[2,0])
        ax3 = sns.lineplot(x=feature , y=target , data=data)
        ax3.set_title('uncertainity w.r.t '+target)

        ax4 = fig.add_subplot(gs[1:,1:3])
        xmin = df.min()
        xmax = df.max()
        ymin = dt.min()
        ymax = dt.max()
        hb = ax4.hexbin(df , dt , gridsize=20 , cmap=pallete)
        ax4.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax4.set_title('Marginal Distribution of '+feature+' and '+target)
        box = ax4.get_position()
        spacing = 0.15
        rect_histx = [box.x0, box.y0-spacing, box.x1/2, 0.03]
        # rect_histy = [box.x0 + box.x1 + spacing, box.y0, 0.2, box.y1]
        ax_c = plt.axes(rect_histx)
        cb = fig.colorbar(hb ,cax=ax_c, orientation='horizontal')
        ax5 = fig.add_subplot(gs[0,1:3])
        sns.distplot(df ,kde=False , rug=True, ax=ax5 )
        ax5.set(xlim=(xmin , xmax))
        ax5.set_title('distribution of '+feature)
        ax5.set_xlabel("")
        ax5.set_xticks([])

        ax6 = fig.add_subplot(gs[1:,3])
        sns.distplot(dt ,ax=ax6,kde=False , rug=True, vertical=True )
        ax6.set(ylim=(ymin , ymax))
        ax6.set_title('distribution of '+target)
        ax6.set_ylabel('')
        ax6.set_yticks([])

        ax7 = fig.add_subplot(gs[0,3])
        sns.kdeplot(df , dt , shade=True , ax=ax7)
        ax7.set_title('pairwise KDE contour \n '+feature+' and '+target)

    elif not continous and target_continous:
        print(f'number of unique classes for {feature} are {data[feature].nunique()} = {data[feature].unique()}')
        print(f'value counts of each class for feature({feature}) \n {data[feature].value_counts()}')
        b_classes = data[feature].unique()
        print('Distribution accross target : '+target)
        df = data.loc[data[feature]==b_classes[0]]
        emp = {}
        emp.update({'counts_prop':len(df)/len(data)})
        emp.update({'mean':df[target].mean()})
        emp.update({'median':df[target].median()})
        emp.update({'std':df[target].std()})
        emp.update({'Q1':df[target].quantile(0.25)})
        emp.update({'Q3':df[target].quantile(0.75)})
        emp.update({'min':df[target].min()})
        emp.update({'max':df[target].max()})
        da = pd.DataFrame(emp.values() ,columns=[b_classes[0]] ,index=emp.keys())
        for b in b_classes[1:]:
            df = data.loc[data[feature]==b]
            emp = {}
            emp.update({'counts_prop':len(df)/len(data)})
            emp.update({'mean':df[target].mean()})
            emp.update({'median':df[target].median()})
            emp.update({'std':df[target].std()})
            emp.update({'Q1':df[target].quantile(0.25)})
            emp.update({'Q3':df[target].quantile(0.75)})
            emp.update({'min':df[target].min()})
            emp.update({'max':df[target].max()})
            dx = pd.DataFrame(emp.values() ,columns=[b] ,index=emp.keys())
            da = pd.concat([da,dx] , axis=1)
        fig = plt.figure(figsize=(10, 14))
        gs = GridSpec(3,2)

        fig.add_subplot(gs[0,0])
        ax1 = sns.countplot(y=feature, data=data ,orient='h')
        ax1.set_title('Frequency of each class of '+feature)

        ax2 = fig.add_subplot(gs[0,1])
        for b in b_classes:
            df = data.loc[data[feature]==b]
            sns.kdeplot(df[target], ax = ax2)
        ax2.set_title('KDE distribution of '+target+" \n for each class of "+feature)

        fig.add_subplot(gs[1,0:])
        ax3 = sns.stripplot(x=target , y=feature , data=data , jitter=jitter )
        ax3.set_title('Class wise distribution for '+target)

        fig.add_subplot(gs[2,0:])
        ax4 = sns.boxplot(x=target , y = feature , data=data , width=0.2)
        ax4.set_title('Boxplot visualization of distribution of '+target+" for each of "+feature)

    else:
        print(f'number of unique classes for {feature} are {data[feature].nunique()} = {data[feature].unique()}')
        print(f'value counts of each class for feature({feature}) \n {data[feature].value_counts()}')
        n_classes = data[target].unique()
        b_classes = data[feature].unique()
        print('Distribution accross target : '+target)
        da = data.loc[data[target]==n_classes[0]]
        da = da[feature].value_counts()
        da = pd.DataFrame(da)
        da = da/da.sum()
        da = da.rename(columns={da.columns[0]:n_classes[0]})
        for c in n_classes[1:]:
            x = data.loc[data[target]==c]
            x = x[feature].value_counts()
            x = x/x.sum()
            da[c] = x

        fig = plt.figure(figsize=(12,5))
        gs = GridSpec(1,3)
        fig.add_subplot(gs[0,0])
        ax = sns.countplot(x=feature , data=data , orient='h')
        ax.set_title('Distribution of ' + feature)
        fig.autofmt_xdate()
        
        fig.add_subplot(gs[0,1:])
        ax2 = sns.countplot(x=target ,hue=feature, data=data)
        ax2.set_title('Distribution of '+feature+' w.r.t. churn')
        plt.legend(loc='best')
    
    fig.tight_layout(pad=2)
    if save_fig:
        plt.savefig('plot.png' , pad_inches=0.2,paper_size='a3')
    
    return da