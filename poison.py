#%%
import pandas as pd
import numpy as np

#データの取り込み tsv形式のためdelimiterで処理する
train = pd.read_csv(r'train.tsv', delimiter='\t')
test = pd.read_csv(r'test.tsv' , delimiter='\t')

# %%
#データサイズの確認
print("Train Data size", train.shape)
print("Test Data size", test.shape)

# %%
train.head()

# %%
#データの中身を確認
train.info()

# %%
#基本統計の確認
train.describe()

# %%
#オブジェクト型の説明変数を抜き出して内容を確認する
cat_cols = [col for col in train.columns if train[col].dtype in ['O']]

print(cat_cols)

#カラム数が多くても省略されないようにする
pd.set_option('display.max_columns', 100)
train[cat_cols].describe()
#欠損値が無いことを確認

# %%

new_train = train.drop(['id'], axis=1)

#説明変数と目的変数を分離する
test_arrange = new_train.loc[:,['Y']]
train_arrange = new_train.drop(['Y'], axis = 1)


#ラベルエンコーディング
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

print(train_arrange.columns)
print(test_arrange.columns)
# %%
#ラベル エンコーディング
#LE.fit_transform(train_arrange['cap-shape', 'cap-surface','cap-color', 'bruises' , 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape','stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring' , 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'].values)

for col in train_arrange.columns: 
    if train[col].dtype in ['O']:
        print(col)
        LE.fit_transform(train_arrange[col].values)
        #データ変換
        train_arrange[col] = LE.fit_transform(train_arrange[col].values)


# %%
train_arrange.head()

train_arrange.info()
# %%

#One Hot Encoding
test_arrange = pd.get_dummies(test_arrange, drop_first=True, columns=['Y'])


#クロスバリデーション
import xgboost as xgb

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_valid, y_train, y_valid = train_test_split(train_arrange, test_arrange, random_state=42, test_size = 0.1, shuffle=True)

train_set = xgb.DMatrix(X_train, y_train)
valid_set = xgb.DMatrix(X_valid, y_valid)

params = {
    'max_depth': 3, 
    'eta': 0.9, 
    'objective': 'multi:softmax', 
    'num_class': 3
}

model = xgb.train(
    params = params,
    dtrain = train_set,
    evals = [(train_set, "train"), (valid_set, "valid")]
)

pred = model.predict(xgb.DMatrix(X_valid))
from sklearn.metrics import accuracy_score

score = accuracy_score(y_valid, pred)
print('score:{0:.4f}'.format(score))

#%%

#判別するデータ
test_data = pd.read_csv(r'test.tsv',delimiter='\t')
test_data.head()
test_data.info()

# %%
test_data_arrange = test_data.drop(['id'], axis=1)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for col in test_data_arrange.columns: 
    if test_data_arrange[col].dtype in ['O']:
        print(col)
        LE.fit_transform(test_data_arrange[col].values)
        #データ変換
        test_data_arrange[col] = LE.fit_transform(test_data_arrange[col].values)

# %%

pred = model.predict(xgb.DMatrix(test_data_arrange))
# %%
id = pd.DataFrame(test_data['id'], columns=["id"])
result = pd.DataFrame(pred, columns=['Y'], dtype=np.uint8)

marge_data = pd.concat([id, result], axis=1, ignore_index=True)

marge_data.columns= ['id', 'Y']

marge_data['Y'] = marge_data['Y'].replace([1,0] , ["p", "e"])


marge_data.to_csv(r"D:\\ML_WORKSPACE\\signate\\105__毒キノコ\\result.csv", index=None, header=True)

# %%
