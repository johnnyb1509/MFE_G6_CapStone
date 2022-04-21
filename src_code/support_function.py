import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        # print(datax.values)
        # print("=================")
        # print(datay.shift(lag).values)
        return datax.corr(datay.shift(lag))
    
def timeLaggedCrossCorr(frame, col_s1, col_s2):
    """Time Lagged Cross Correlation

    Args:
        frame (pd.DataFrame): 
        col_s1 (str): column name - the variable that fix
        col_s2 (_type_): column name - the variable that shift in range of offsets
    """
    d1 = frame[col_s1]
    d2 = frame[col_s2]
    fps = 16
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(fps),int(fps))]
    offset = np.ceil(len(rs)/2)-np.argmax(rs)
    f,ax=plt.subplots(figsize=(12,6))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs)/2.75),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    ax.axvline(np.argmin(rs),color='g',linestyle='--',label='Trough synchrony')
    ax.set(title=f'Time Lagged Cross Correlation\nOffset = {offset} frames\n{col_s1} leads <> {col_s2} leads',ylim=[-0.75,0.75],xlim=[0,16], xlabel='Offset',ylabel='Pearson r')
    ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
    ax.set_xticklabels([-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,-12]);
    plt.legend()
    plt.show();
    return

def WindowTLCC(frame, col_s1, col_s2):
    """Windowed Time Lagged Cross Corelation and Rolling variation

    Args:
        frame (pd.DataFrame): 
        col_s1 (str): column name - the variable that fix
        col_s2 (_type_): column name - the variable that shift in range of offsets
    """
    # Windowed time lagged cross correlation
    fps = 12
    no_splits = 4
    samples_per_split = int(frame.shape[0]/no_splits)
    rss=[]
    for t in range(0, no_splits):
        d1 = frame[col_s1].iloc[(t)*samples_per_split:(t+1)*samples_per_split]
        d2 = frame[col_s2].iloc[(t)*samples_per_split:(t+1)*samples_per_split]
        rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(fps),int(fps+1))]
        rss.append(rs)

    rss = pd.DataFrame(rss)

    f,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(rss,cmap='RdBu_r',ax=ax,vmin=-1, vmax=1)
    ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,15], xlabel='Offset',ylabel='Epochs')
    ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
    ax.set_xticklabels([-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,-12])
    plt.show();

    # Rolling window time lagged cross correlation
    fps = 12
    window_size = 24 #samples
    t_start = 0
    t_end = t_start + window_size
    step_size = 12
    rss=[]
    while t_end < frame.shape[0]:
        d1 = frame[col_s1].iloc[t_start:t_end]
        d2 = frame[col_s2].iloc[t_start:t_end]
        rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(fps),int(fps+1))]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    f,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(rss,cmap='RdBu_r',ax=ax,vmin=-1, vmax=1)
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation',xlim=[0,15], xlabel='Offset',ylabel='Epochs')
    ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
    ax.set_xticklabels([-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,-12])
    plt.show();
    return

def plot_windowCorrelation(frame, name_x, name_y, name_plot = None, r_window_size = 8):
    """Calcualte and ploting wolling correlation

    Args:
        frame (_type_): _description_
        name_x (_type_): _description_
        name_y (_type_): _description_
        name_plot (str): Name of column want to plotwiht "name_x"
        r_window_size (int, optional): rolling distance. Defaults to 8.
    """
    rolling_r = frame[name_x].rolling(window=r_window_size).corr(frame[name_y])
    f,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
    if name_plot is None:
        frame[[name_x, name_y]].plot(ax=ax[0])
    else:
        frame[[name_x, name_plot]].plot(ax=ax[0])
    ax[0].set(xlabel='Time Series', ylabel='Variable Change')
    # ax[0].set_ylim(bottom = min(frame[name_plot].values)*1.5, top = max(frame[name_x].values)*1.5)
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Correlation',ylabel='Pearson Correlation')
    ax[1].axhline(color='k',linestyle='--',label='Center')
    ax[1].set_ylim(bottom = -1, top = 1)
    plt.suptitle(f"Window Correlation between {name_x} vs. {name_y} | Rolling = {r_window_size}")
    plt.show();
    
    rolling_r = rolling_r.to_frame(name = f'{name_x}_vs_{name_y}')
    return rolling_r

def qq_plot(frame, figsize = (25,70)):
    """Generate qq plot for each column in the Frame

    Args:
        frame (pandas Frame): 
        figsize (tuple, optional): size of figure. Defaults to (25,70).
    """
    from scipy.stats import probplot

    n_cols = 3
    n_rows = round(len(frame.columns)/n_cols + 0.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    c = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes [row, col]
            probplot(frame[frame.columns[c]], dist = 'norm', plot = ax)
            ax.set_title(frame.columns[c])

            if c >= len(frame.columns)-1:
                print('Finish ploting the QQ-plot')
                break
            else:
                c += 1
    print(f'Ploted {c+1}/{len(frame.columns)} variables')
    plt.show()
    return

def box_plot_corr(frame_correlation, n_box = 3, title = None):
    import math
    columns = list(frame_correlation.columns)
    
    # Generate list of <n_box> in 1 plot
    n_list = int(math.ceil(len(columns)/n_box))
    list_comb = list()
    cache_list = list()
    for idx in range(len(columns)):
        cache_list.append(columns[idx])
        if len(cache_list) == n_box or columns[idx] == columns[-1]:
            list_comb.append(cache_list)
            cache_list = list()

    assert len(list_comb) == n_list
    
    # Plot boxplot
    for list_ in list_comb:
        ax = frame_correlation[list_].plot(kind = "box", figsize=(15,8))
        ax.set_ylim(top = 1, bottom = -1)
        if title is None:
            ax.set_title("Correlation Character between Variables", fontsize=14)
        else:
            ax.set_title(title, fontsize=14)
        ax.set_ylabel('Correlation')
        ax.axhline(color='r',linestyle='--',label='Center')
        plt.show();
    return

def plotOnEvent(frame, columns):
    """plot graph on defined event

    Args:
        frame (DataFrame): frame to plot
        columns (list): list of column to plot
    """
    from matplotlib.dates import date2num
    from datetime import datetime
    test = frame.copy()
    test.index = test.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(test.index,test[columns])
    # test.plot(figsize = (12,8), ax = ax)
    ax.axvspan(date2num(datetime(2008,1,12)), date2num(datetime(2010,6,1)), # datetime(2007,1,12)), date2num(datetime(2009,6,1)
            label="2009 Recession", color="gray", alpha=0.3)
    ax.axvspan(date2num(datetime(2011,5,1)), date2num(datetime(2012,7,1)),  # datetime(2010,5,1)), date2num(datetime(2011,7,1)
            label="Sovereign Debt in Europe", color="gray", alpha=0.3)
    ax.axvspan(date2num(datetime(2016,1,1)), date2num(datetime(2016,12,31)), # datetime(2015,1,1)), date2num(datetime(2015,12,31)
            label="China capital ouflow", color="gray", alpha=0.3)
    ax.axvspan(date2num(datetime(2017,10,1)), date2num(datetime(2017,11,3)), # datetime(2016,10,1)), date2num(datetime(2016,11,3)
            label="OPEC cut oil supply", color="gray", alpha=0.3)
    ax.axvspan(date2num(datetime(2018,1,1)), date2num(datetime(2019,1,1)), # datetime(2017,1,1)), date2num(datetime(2018,1,1)
            label="DXY drop", color="gray", alpha=0.3)
    ax.axvspan(date2num(datetime(2019,7,6)), date2num(datetime(2021,1,13)), # datetime(2018,7,6)), date2num(datetime(2020,1,13)
            label="2009 Recession", color="gray", alpha=0.3)
    ax.axvspan(date2num(datetime(2020,12,31)), date2num(datetime(2021,12,31)), # datetime(2019,12,31)), date2num(datetime(2021,12,31)
            label="2009 Recession", color="gray", alpha=0.3)
    ax.legend(columns)
    plt.show();
    return

def k_cluster(dataframe, n_cluster = 3):
    """For cluster the covariance or correlation matrix

    Args:
        dataframe (pandas df): correlation matrix

    Returns:
        data frame: [description]
    """
    cluster_etf = []
    from sklearn.cluster import KMeans
    # Clustering
    kmeans = KMeans(n_clusters = n_cluster, random_state = 42)
    label = kmeans.fit_predict(dataframe.values)
    u_labels = np.unique(label)
    
    frame = pd.DataFrame(np.array([np.array(kmeans.labels_), np.array(dataframe.index.values)]).T, columns = ['label', 'components'])
    
    # print group
    for g in np.unique(kmeans.labels_):
        print(frame.groupby('label').get_group(g))
    
    return kmeans, frame