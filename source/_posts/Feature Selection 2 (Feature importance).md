---
title:  Feature Selection 2
date: 2019-04-25
categories:
    - 特征工程
tags:
    - Feature Selection
    - Feature Importances
mathjax: true
---
# Feature importance

## RF feature importance

All tree-based models have `feature_importances_`, like RandomForestClassifier (xgboost, lightgbm). For classification, it is typically either [Gini impurity](http://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) or [information gain/entropy](http://en.wikipedia.org/wiki/Information_gain_in_decision_trees) and for regression trees it is [variance](http://en.wikipedia.org/wiki/Variance). Thus when training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure. [2]

There are a few things to keep in mind when using the impurity based ranking:  [2]

1. **feature selection based on impurity reduction is biased towards preferring variables with more categories** (see [Bias in random forest variable importance measures](http://link.springer.com/article/10.1186%2F1471-2105-8-25)). 

2.  when the dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be used as the predictor, with no concrete preference of one over the others. But once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature. As a consequence, they will have a lower reported importance. 



## Boruta

[3]  shadow feature



## Permutation Importance (Mean decrease accuracy)

The general idea is to permute the values of each feature and measure how much the permutation decreases the accuracy of the model.

1. Get a **trained** model
2. Shuffle the values in a **single** column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
3. Return the data to the original order (undoing the shuffle from step 2.) Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.

**Permutation importance is calculated after a model has been fitted.** So we won't change the model or change what predictions we'd get for a given value of height, sock-count, etc. [1]

#### Code example [1]

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

The values towards the top are the most important features, and those towards the bottom matter least.

#### Code example [2]

```python
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
X = boston["data"]
Y = boston["target"]
rf = RandomForestRegressor()
scores = defaultdict(list)

#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
# Outputs: Features sorted by their score
# [(0.7276, 'LSTAT'), (0.5675, 'RM'), (0.0867, 'DIS'), (0.0407, 'NOX'), (0.0351, 'CRIM'), (0.0233, 'PTRATIO'), (0.0168, 'TAX'), (0.0122, 'AGE'), (0.005, 'B'), (0.0048, 'INDUS'), (0.0043, 'RAD'), (0.0004, 'ZN'), (0.0001, 'CHAS')]
```

`LSTAT` and `RM` are two features that strongly impact model performance: permuting them decreases model performance by ~73% and ~57% respectively. Keep in mind that these measurements are made only after the model has been trained (and is depending) on all of these features. 



## Null Importance (target permutation) [4]

The [original paper](<https://academic.oup.com/bioinformatics/article/26/10/1340/193348>). 较Boruta的优势：Boruta只适合用sklearn的RandomForest、需要填充缺失值、特征数量扩大一倍也会更加消耗资源。

1. 将label的数据顺序打乱，使用树模型训练，得到每个特征对于label顺序打乱之后的feature importance；

2. 重复以上操作多次（e.g. 80次），每个特征都有80个label随机打乱顺序后得到的feature importance，这80个feature importance构成一个分布；
3. 对于每个特征，不打乱顺序的actual feature importance在这个分布下的显著性水平

为什么叫Null importance呢？考虑原假设（Null hypothesis）：特征与label不相关。随机打乱label顺序，特征的feature importance的分布称之为Null importance distribution。

examples: [4]

![1](https://www.kaggleusercontent.com/kf/4065111/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..1rxhm_IaEkkmRrXju0mIUA.eLWISm6ZTaq6XGscOq4aX9YtYP8LWXb0AUeb9oJJckKKYBuPZL-epkXonVD6V78abGSwYW0XOXzsgb70ns_nwR-gGyT1Jfek77FhoHkq4wCNF39aezzYX-QFsJhd4AWlA_X_6zazpJjLXJS3OKo_kixQzt-1VeDw6n7G8xcAtFs.5FPxMnhW2ZwP6LCHjf8bNg/__results___files/__results___15_0.png)

![2](https://www.kaggleusercontent.com/kf/4065111/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..1rxhm_IaEkkmRrXju0mIUA.eLWISm6ZTaq6XGscOq4aX9YtYP8LWXb0AUeb9oJJckKKYBuPZL-epkXonVD6V78abGSwYW0XOXzsgb70ns_nwR-gGyT1Jfek77FhoHkq4wCNF39aezzYX-QFsJhd4AWlA_X_6zazpJjLXJS3OKo_kixQzt-1VeDw6n7G8xcAtFs.5FPxMnhW2ZwP6LCHjf8bNg/__results___files/__results___16_0.png)

蓝色hist为label顺序打乱之后的feature impotance distribution，红色线为实际的feature importance值。

> Under the null hypothesis and normal distribution if for a feature the red actual importance is within the blue distribution then chances are the feature is not correlated with the target. If it's within the 5% right part of the blue distribution or outside then it's correlated.

第一个特征的Gain importance落在了分布之内，显著性水平肯定>0.05，所以可以认为这个特征与实际的label不相关。第二个特征的actual feature importance与null feature importance distribution相距甚远，所以认为这个特征与actual label有相关性。

Score features：$$score = \log(10^{-10} + \frac{actual~imp}{1+percentile(null~imp,75)})$$，加入1e-10是因为`np.log(0)`会报错，同理分母+1也是防止报错。

Assess correlation to the target：$$corr\_score = \frac{|null<act|}{null}*100\%$$. Build this metric to show hwo far the actual importance is from the noise (null importance) distribution. 通过设定不同的阈值做feature selection，得到不同的feature集合，再用这些feature训练lgb模型，通过cv的结果来比较不同的feature集合：

```python
def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': .1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 13,
        'n_jobs': 4,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc'
    }
    
    # Fit the model
    hist = lgb.cv(
        params=lgb_params, 
        train_set=dtrain, 
        num_boost_round=2000,
        categorical_feature=cat_feats,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=17
    )
    # Return the last mean / std values 
    return hist['auc-mean'][-1], hist['auc-stdv'][-1]

for threshold in [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]:
    split_feats = [_f for _f, _score, _ in corr_scores if _score >= threshold]
    split_cat_feats = [_f for _f, _score, _ in corr_scores if (_score >= threshold) & (_f in categorical_feats)]
    gain_feats = [_f for _f, _, _score in corr_scores if _score >= threshold]
    gain_cat_feats = [_f for _f, _, _score in corr_scores if (_score >= threshold) & (_f in categorical_feats)]
                                                                                             
    print('Results for threshold %3d' % threshold)
    split_results = score_feature_selection(df=data, train_features=split_feats, cat_feats=split_cat_feats, target=data['TARGET'])
    print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
    gain_results = score_feature_selection(df=data, train_features=gain_feats, cat_feats=gain_cat_feats, target=data['TARGET'])
    print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
```



## Reference:

1. https://www.kaggle.com/dansbecker/permutation-importance

2. [Selecting good features – Part III: random forests](https://blog.datadive.net/selecting-good-features-part-iii-random-forests/)

3. [Feature Importance Measures for Tree Models — Part I](https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3)

   - https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features

   - https://www.kaggle.com/ogrellier/feature-selection-target-permutations (use target permutation instead of feature permutation)

   - http://danielhomola.com/2015/05/08/borutapy-an-all-relevant-feature-selection-method/


4. [Feature Selection with Null Importances](<https://www.kaggle.com/ogrellier/feature-selection-with-null-importances>)

   Note: convert all category features to `.astype('category')`, not using `pd.factorize`. Because the latter will convert all unknown or NaN values to -1, but lgb uses `LabelEncoder` which cannot handle negative values.