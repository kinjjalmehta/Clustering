from os import link, sep
from numpy.lib.function_base import quantile
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt
'''url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
res = requests.get(url, allow_redirects=True)
with open('covid_dataset.csv','wb') as file:
    file.write(res.content)'''


data_original = pd.read_csv('covid_dataset.csv') 
#print(len(data_original))
data = data_original.loc[data_original["Population"] != 0 ]  # remove rows whose population is 0
grouped_data = data.groupby(['Province_State']) # group states togethers
grouped_and_summed =grouped_data.sum()
grouped_and_summed = grouped_and_summed.reset_index()

# remove places that are not US states
grouped_and_summed.drop(grouped_and_summed.loc[grouped_and_summed['Province_State']=='Northern Mariana Islands'].index,inplace=True)
grouped_and_summed.drop(grouped_and_summed.loc[grouped_and_summed['Province_State']=='Guam'].index,inplace=True)
grouped_and_summed.drop(grouped_and_summed.loc[grouped_and_summed['Province_State']=='Virgin Islands'].index,inplace=True)
grouped_and_summed.drop(grouped_and_summed.loc[grouped_and_summed['Province_State']=='Puerto Rico'].index,inplace=True)
grouped_and_summed.drop(grouped_and_summed.loc[grouped_and_summed['Province_State']=='District of Columbia'].index,inplace=True)
grouped_and_summed.drop(grouped_and_summed.loc[grouped_and_summed['Province_State']=='American Samoa'].index,inplace=True)

# Time Series Analysis

wisc_row = grouped_and_summed[grouped_and_summed['Province_State'] == 'Wisconsin'].loc[:,'1/22/20':'8/22/20']
cali_row = grouped_and_summed[grouped_and_summed['Province_State'] == 'California'].loc[:,'1/22/20':'8/22/20']

wisc_row_cumulative = wisc_row.cumsum(axis=1)   # cumulative time series of Wisconsin and California
cali_row_cumulative = cali_row.cumsum(axis=1)

wisc_row_diff = wisc_row.diff(axis=1).iloc[:,1:]  # differenced time series of Wisconsin and California
cali_row_diff = cali_row.diff(axis=1).iloc[:, 1:].astype(int)

print(wisc_row_diff.values.tolist())
#print(np.asarray(cali_row_diff).tolist())


# parameters -> mean, standard deviation, median, 50th quantile 

target = grouped_and_summed.loc[:,'1/22/20':'8/22/20']
target_reverse=target[target.columns[::-1]]

target['mean'] = target.mean(axis=1)
target['sd'] = target.std(axis=1)
target['median'] = target.median(axis=1)
target['max'] = target.max(axis=1)
target['quant'] = target.quantile(0.5,axis=1)

# a separate dataframe for the parameters
params_df = target[target.columns[-5:]]

# scale the values
mean_max = params_df['mean'].max()
mean_min = params_df['mean'].min()
for i in params_df.index:
    params_df.at[i,'mean'] = (params_df.at[i,'mean'] - mean_min / mean_max - mean_min) * 100

sd_max = params_df['sd'].max()
sd_min = params_df['sd'].min()
for i in params_df.index:
    params_df.at[i,'sd'] = (params_df.at[i,'sd'] - sd_min / sd_max - sd_min) * 100

median_max = params_df['median'].max()
median_min = params_df['median'].min()
for i in params_df.index:
    params_df.at[i,'median'] = (params_df.at[i,'median'] - median_min / median_max - median_min) * 100

max_max = params_df['max'].max()
max_min = params_df['max'].min()
for i in params_df.index:
    params_df.at[i,'max'] = (params_df.at[i,'max'] - max_min / max_max - max_min) * 100

quant_max = params_df['quant'].max()
quant_min = params_df['quant'].min()
for i in params_df.index:
    params_df.at[i,'quant'] = (params_df.at[i,'quant'] - quant_min / quant_max - quant_min) * 100

params_scaled = params_df.round(2) # round to 2 decimal places

condensed_matrix = spd.pdist(params_scaled, metric='euclidean')
dendogram = sch.dendrogram(sch.linkage(params_scaled, method='single'))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.show()

# finding hierarchial cluster labels

cluster = AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='complete',compute_distances=True)
cluster.fit_predict(params_scaled)
cluster.labels_.tolist()

# finding cluster centers, cluster labels and k-means total distortion.

kmeans = KMeans(n_clusters=7, random_state=12345)
kmeans.fit_transform(params_scaled)
kmeans.labels_.tolist()
kmeans.inertia_     # total distortion

ccenter = kmeans.cluster_centers_
np.around(ccenter, decimals=4)

'''
n = grouped_and_summed.loc[:,'8/22/20']
threshold10 = n / 10
target['growth_rate'] = threshold10

# number of days till 10 times less cases
t = len(target_reverse.columns)
for index in target_reverse.index, threshold10:
    for column in target_reverse:
        if target_reverse[index,column] <= threshold10[index]:
            target[index,'growth_rate'] = 
'''




