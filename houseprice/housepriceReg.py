"""
autoencoder
"""
# import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb  # light gradient boosting forest
import random
# from FeatureMaps import featuremap
from sklearn.preprocessing import LabelEncoder

plt.ion()

data_dir = 'house-prices-advanced-regression-techniques/'
fname_train = 'train.csv'
fname_test  = 'test.csv'

def fmissing(df):
    ''' This function imputes missing values with median for numeric columns
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)

def cats2nums(df):
    # Find the columns of object type along with their column index
    object_cols = list(df.select_dtypes('O').columns)
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))

    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])

# load data
data = pd.read_csv(data_dir+fname_train, index_col='Id')
salep = data.SalePrice
# snstd = (salep - salep.mean())/salep.std()
hfeat = data.drop('SalePrice', axis=1)
catcols = hfeat.select_dtypes('O').columns
catind = []
for col in catcols:
    catind.append(hfeat.columns.get_loc(col))
fmissing(hfeat)
cats2nums(hfeat)
# for k in hfeat.keys():
#     if featuremap[k]:
#         for w,v in featuremap[k].items():
#             hfeat[k][hfeat[k]==w] = v
# hfeat = hfeat.fillna(0)
data_test = pd.read_csv(data_dir+fname_test, index_col='Id')
# for k in data_test.keys():
#     if featuremap[k]:
#         for w,v in featuremap[k].items():
#             data_test[k][data_test[k]==w] = v
# data_test = data_test.fillna(0)
fmissing(data_test)
cats2nums(data_test)
train_data = lgb.Dataset(hfeat[:-300],np.log(salep[:-300]))  # snstd, categorical_feature=catind
valid_data = lgb.Dataset(hfeat[-300:],np.log(salep[-300:]), reference=train_data)  #, categorical_feature=catind
pred_data = lgb.Dataset(data_test)

# param = {'num_leaves': 31, 'objective': 'regression', 'metric': ['auc', 'binary_logloss']}
param = {
    'boosting_type': 'rf',  # gbdt, dart, goss
    'objective': 'regression',
    'metric': {'rmse'}, # 'binary_logloss', 'l2', 'l1', 'mape'
    'num_leaves': 63,
    # 'lambda_l2': .2,
    'learning_rate': 0.01,
    'feature_fraction': 0.2,  #
    'bagging_fraction': 0.5,  # .8
    'bagging_freq': 4,
    'verbose': -1,
}

num_round = 1600
early_stop = 100
n_forests = 10

for i in range(n_forests):
    param['seed'] = i
    bstrf = lgb.train(param, train_data, num_round, valid_sets=valid_data, early_stopping_rounds=early_stop)

    # save model
    bst.save_model('forest_model_0.01_%d.txt' % i, num_iteration=bst.best_iteration)

# validification
errlist = []
vp = []
nt = 0
for i in range(n_forests):
    # load model
    bst = lgb.Booster(model_file='forest_model_0.01_%d.txt' % i)  #
    vprice = np.exp(bst.predict(hfeat[-300:]))
    if len(vp)>0:
        vp += vprice*bst.num_trees()
    else:
        vp = vprice*bst.num_trees()
    nt += bst.num_trees()
    e0 = np.sqrt(((np.log(vprice)-np.log(salep[-300:]))**2).mean())
    errlist.append(e0)

vp /= nt
err = np.sqrt(((np.log(vp)-np.log(salep[-300:]))**2).mean())

# prediction
# for i in range(n_forests):
#     # load model
#     bst = lgb.Booster(model_file='forest_model_0.01_%d.txt' % i)  #
#     pprice = bst.predict(pred_data)
