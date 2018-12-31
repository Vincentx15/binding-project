import pandas as pd
import os
import csv
import csv_builder as cb
import numpy as np
from sklearn.preprocessing import *

"""
PREPARE A DUMMY CSV
"""
# cb.prepare_csv(csv_path)
# for i in range(100):
#     row = ['ligand{}'.format(str(i))]+['True'] + list(np.random.rand(60))
#     with open(csv_path, "a", newline='') as fp:
#         wr = csv.writer(fp, dialect='excel')
#         wr.writerow(row)

# values = df[df.columns[2:]]
# scaler = StandardScaler()
# values[values.loc[:,:]] = scaler.fit_transform(values[values.loc[:,:]])


"""
READ EMBEDDINGS
"""

# data/embeddings/aa2ar/csv_actives_feature=False_pca=False.csv
# name = 'csv_actives_feature=True_pca=False.csv'
# csv_path = os.path.join('../data/embeddings/aa2ar', name)
# df = pd.read_csv(csv_path, header=0)
# df = df[['f2c2m1','f2c2m2','f2c2m3']]
# print(df.head())
#
# name = 'csv_actives_feature=True_pca=True.csv'
# csv_path = os.path.join('../data/embeddings/vgfr2', name)
# df = pd.read_csv(csv_path, header=0)
# df = df[['f2c2m1','f2c2m2','f2c2m3']]
# print(df.head())
#
# name = 'csv_actives_GM.csv'
# csv_path = os.path.join('../data/embeddings/aa2ar', name)
# df = pd.read_csv(csv_path, header=0)
# df = df[['f1c2m1','f1c2m2','f1c2m3']]
# print(df.head())
#
# name = 'EF=0.25.csv'
# csv_path = os.path.join('../data/raw_output_csv', name)
# df = pd.read_csv(csv_path, header=0)
# print(df.describe())
#
# name = 'EF=1.csv'
# csv_path = os.path.join('../data/raw_output_csv', name)
# df = pd.read_csv(csv_path, header=0)
# print(df.describe())
# #

# name = 'EF=1.csv'
# csv_path = os.path.join('../data/raw_output_csv', name)
# df = pd.read_csv(csv_path, header=0)
# print(df.describe())
#
# name = 'GM_EF=0.1.csv'
# csv_path = os.path.join('../data/raw_output_csv', name)
# df = pd.read_csv(csv_path, header=0)
# df.rename(index=str,columns={'feature_pca': 'GM'})
# csv_path_new = csv_path + 'toto'
# df.to_csv(csv_path_new)



"""
READ CSV
"""

# csv_path = '../output_csv/8_EF=0.25.csv'
# df = pd.read_csv(csv_path, header=0)
# print(df)


# SORT IT
# csv_path = '../output_csv/whole_separated.csv'
# df = pd.read_csv(csv_path, header=0, index_col=0)
# df = df.set_index('embedding')
# print(df)
# df = df.sort_values(by=['EF=0.25'])
# print(df)
# df.to_csv('../data/output_csv/whole_separated_sorted.csv')

# csv_path = '../data/output_csv/EF=0.25.csv'
# df = pd.read_csv(csv_path, header=0)
# df = df.set_index('structure')
# col = df[df.index.str.match('aa2ar')]
# print(col)


"""
SCALE IT
"""

# def scale_one (name):
#     csv_dir = '../data/embeddings/aa2ar'
#     csv_path = os.path.join(csv_dir, name)
#     df = pd.read_csv(csv_path, header=0)
#     print(df.describe())
#
#     scaler = StandardScaler()
#     df[df.columns[2:]] = scaler.fit_transform(df[df.columns[2:]])
#     df = df[df.columns[1:]]
#     print(df.describe())
#
#     csv_dir = '../data/embeddings/test_scaling'
#     csv_path = os.path.join(csv_dir, name)
#     # df.to_csv(csv_path)
#
# def scale_both(name):
#     """
#     :param name: just the name of the embedding
#     :return:
#     """
#     name_actives = 'csv_actives_'+name+'.csv'
#     scale_one(name_actives)
#     name_decoys = 'csv_decoys_'+name+'.csv'
#     scale_one(name_decoys)
#
#
# scale_both('feature=True_pca=True')

"""
MERGE RESULT CSV from ../output_csv to data/output_csv
"""


# csv_dir = '../data/raw_output_csv'
# for csv in os.listdir(csv_dir):
#     csv_path = os.path.join(csv_dir, csv)
#     df = pd.read_csv(csv_path, header=0)
#     list1 = list(df['structure'])
#     print (csv)
#
# filtre = 'EF=1'
# csv_dir = '../data/raw_output_csv'
#
# list_df = []
# for csv in os.listdir(csv_dir):
#     if csv.find(filtre) >= 0:
#         csv_path = os.path.join(csv_dir, csv)
#         df = pd.read_csv(csv_path, header=0)
#         list_df.append(df)
#
# df = list_df.pop()
# while len(list_df):
#     df2 = list_df.pop()
#     df = pd.merge(df, df2, on=['structure'])
#
# csv_pathway = '../data/output_csv/' + filtre + '.csv'
# df.to_csv(csv_pathway)


"""
AGGREGATE RESULTS FOR ALL EF
"""
# filtre = 'EF'
# csv_dir = '../data/output_csv'
#
# list_df = []
# header = []
# for csv in os.listdir(csv_dir):
#     if csv.find(filtre) >= 0:
#         csv_path = os.path.join(csv_dir, csv)
#         df = pd.read_csv(csv_path, header=0)
#         df = df[df.columns[1:]]
#         df = df.mean(axis=0)
#         list_df.append(df)
#         colname = csv.replace('.csv','')
#         header.append(colname)
# res = pd.concat(list_df, axis=1)
# res.columns = header
# print(res)
#
# name = 'whole.csv'
# output_path = os.path.join(csv_dir, name)
# res.to_csv(output_path)


"""
TEST DIFFERENT SIMILARITY MEASURES
"""

# import EF_computer as ec
#
# name = 'csv_actives_feature=True_pca=False.csv'
# csv_path = os.path.join('../data/embeddings/aa2ar', name)
# df = pd.read_csv(csv_path, header=0, nrows=100)
#
# def distance_matrix1(x, y):
#     """
#     copied from scipy documentation
#     """
#     x = np.asarray(x)
#     m, k = x.shape
#     y = np.asarray(y)
#     n, kk = y.shape
#     result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
#     for i in range(m):
#         result[i, :] = [ec.cosine(x[i], y_j) for y_j in y]
#     return result
#
# def distance_matrix2(x, y):
#     """
#     copied from scipy documentation
#     """
#     x = np.asarray(x)
#     m, k = x.shape
#     y = np.asarray(y)
#     n, kk = y.shape
#     result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
#     for i in range(m):
#         result[i, :] = [ec.usfr_similarity(x[i], y_j) for y_j in y]
#     return result
#
# def distance_matrix3(x, y):
#     """
#     copied from scipy documentation
#     """
#     x = np.asarray(x)
#     m, k = x.shape
#     y = np.asarray(y)
#     n, kk = y.shape
#     result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
#     for i in range(m):
#         result[i, :] = [ec.super_similarity(x[i], y_j) for y_j in y]
#     return result
#
# name = 'csv_actives_feature=True_pca=False.csv'
# csv_path = os.path.join('../data/embeddings/aa2ar', name)
# df = pd.read_csv(csv_path, header=0, nrows=100)
# values = df[df.columns[2:]]
#
# actives_dist = distance_matrix1(values, values)
# print(np.std(actives_dist))
#
# actives_dist = distance_matrix2(values, values)
# print(np.std(actives_dist))
#
# actives_dist = distance_matrix3(values, values)
# print(np.std(actives_dist))

"""
Split the 'method in categorical fields
"""

# csv_path = '../data/output_csv/whole.csv'
# df = pd.read_csv(csv_path, header=0)
#
# def label_metrique(row):
#     row = row['method']
#     if row.find('manhattan') >= 0:
#         return 'manhattan'
#     if row.find('cosine') >= 0:
#         return 'cosine'
#     if row.find('mixed') >= 0:
#         return 'mixed'
#     return 0
#
# def label_methode(row):
#     row = row['method']
#     return row[:2]
#
#
# df['embedding'] = df.apply(lambda row: label_methode(row),axis=1)
# df['metrique'] = df.apply(lambda row: label_metrique(row),axis=1)
# df = df.drop(['method'],axis=1)
#
# print(df)
