---
title: Feature Selection 1
date: 2019-04-25
categories:
    - 特征工程
tags:
    - Feature Selection
mathjax: true
---



特征选择是特征工程里的一个重要问题，目的是找到最优特征子集。**减少特征个数**，减少运行时间，提高模型精确度；**更好的理解特征**，及其与label之间的相关性。

- Filter（过滤法）：按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。注意，过滤法不能减弱特征间的共线性。
- Wrapper（包装法）：根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
- Embedded（嵌入法）：先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小排序选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

## Filter特征选择(Univariate selection)

针对样本的每个特征$$x_i~(i=1,\dots,n)$$，计算$$x_i$$与label标签$$y$$的信息量$$S(i)$$，得到$$n$$个结果，选择前$$k$$个信息量最大的特征。即选取与$$y$$关联最密切的一些特征$$x_i$$。下面介绍几种度量$$S(i)$$的方法：

1. Pearson相关系数

   衡量变量之间的线性相关性(linear correlation)，结果取值为$$[-1,1]$$，-1表示完全负相关，+1表示完全正相关，0表示没有**线性**相关。

   简单，计算速度快；但只对线性关系敏感，如果关系是非线性的，即使两个变量有关联，Pearson相关性也可能接近0。scipy的pearsonr方法能计算相关系数和p-value[2], roughly showing the probability of an uncorrelated system creating a correlation value of this magnitude. The p-value is high meaning that it is very likely to observe such correlation on a dataset of this size purely by chance[6]：

   ```python
   import numpy as np
   from scipy.stats import pearsonr
   
   np.random.seed(0)
   size = 300
   x = np.random.normal(0, 1, size)
   print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
   print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))
   # output: (0.718248, 7.324017e-49), (0.057964, 0.317009)
   ```

   类似的，在sklearn中针对回归问题有`f_regression`函数，测量一组变量与label的线性关系的p-value[6]

   Relying only on the correlation value on interpreting the relationship of two variables can be highly misleading, so it is always worth plotting the data[6]

2. **互信息和最大信息系数** Mutual information and maximal information coefficient (MIC)

   MI评价自变量与因变量的相关性。当$$x_i$$为0/1取值时，$$MI(x_i,y) = \sum\limits_{x_i\in\{0,1\}}\sum\limits_{y\in\{0,1\}}p(x_i,y)\log\frac{p(x_i,y)}{p(x_i)p(y)}$$，同理也很容易推广到多个离散值情形。可以发现MI衡量的是$$x_i$$和$$y$$的独立性，如果两者独立，MI=0，即$$x_i$$和$$y$$不相关，可以去除$$x_i$$；反之两者相关，MI会很大。

   MI的缺点：不属于度量方式，无法归一化；无法计算连续值特征，通常需要先离散化，但对离散化方式很敏感。

   MIC解决MI的缺点：首先，寻找最优的离散化方式；然后，把MI变成一种度量方式，区间为$$[0,1]$$
   ```python
   from minepy import MINE
   m = MINE()
   x = np.random.uniform(-1, 1, 10000)
   m.compute_score(x, x**2)
   print m.mic() # output: 1, the maximum
   ```
3. Distance Correlation

   Pearson相较MIC或者Distance correlation的优势：1. 计算速度快；2. correlaiton的取值区间是[-1,1]，体现正负相关性

4. 卡方验证（**常用**）

   基于频率分布来检验分类变量间的相关性。假设自变量有N种取值，因变量有M种取值，自变量等于i且因变量等于j的样本频数的观察值与期望的差距：$$\chi^2 = \sum\frac{(A-E)^2}{E}$$.

   ```python
   from sklearn.datasets import load_iris
   from sklearn.feature_selection import SelectKBest
   from sklearn.feature_selection import chi2
   iris = load_iris()
   X, y = iris.data, iris.target
   #选择K个最好的特征，返回选择特征后的数据
   X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
   # Output:  X.shape = (150,4), X_new.shape = (150,2)
   ```

5. 基于模型的特征排序：

   直接用你要用的模型，对每个**单独**特征和标签$$y$$建立模型。假设此特征和标签的关系是非线形的，可用tree based模型，因为他们适合非线形关系的模型，但要注意防止过拟合，树的深度不要大，并运用交叉验证。
   ```python
   from sklearn.cross_validation import cross_val_score, ShuffleSplit
   from sklearn.datasets import load_boston
   from sklearn.ensemble import RandomForestRegressor
   
   boston = load_boston()
   X = boston["data"]
   Y = boston["target"]
   names = boston["feature_names"]
   
   rf = RandomForestRegressor(n_estimators=20, max_depth=4)
   scores = []
   for i in range(X.shape[1]):
        #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
        #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
        score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                 cv=ShuffleSplit(len(X), 3, .3))
        scores.append((round(np.mean(score), 3), names[i]))
   
   #打印出各个特征所对应的得分
   print(sorted(scores, reverse=True))
   ```

6. Variance Threshold

   但这种方法不需要度量特征$$x_i$$和标签$$y$$的关系。计算各个特征的方差，然后根据阈值选择方差大于阈值的特征。

   ```python
   from sklearn.feature_selection import VarianceThreshold
   X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
   sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
   print(sel.fit_transform(X))
   ```


7. LDA：考察特征的线性组合能否区分一个分类变量
8. ANOVA：原理和LDA类似只是它常用于特征是分类变量，响应变量是连续变量的情况下，它提供了不同组的均值是否相同的统计量

## Wrapper特征选择

在确定模型之后，不断的使用不同的特征组合来测试模型的表现，一般选用普遍效果较好的算法，如RF，SVM，kNN等。

- 前向搜索：每次从未选中的特征集合中选出一个加入，直到达到阈值或n为止

- 后向搜索：每一步删除一个特征

- 递归特征消除法RFE [9]：使用一个模型进行多轮训练，每轮训练后消除一个或多个重要性最低的特征，再基于新特征进行下一轮训练。sklearn中的RFE只能使用带有`coef_`或者`feature_importances_`的模型（所以SVM只能使用默认的liear核，不能使用rbf核）

  RFE明确指定选出几个特征。但使用回归模型时没有正则化会导致模型不稳定，回归模型推荐使用ridge回归。
  ```python
  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LogisticRegression
  #递归特征消除法，返回特征选择后的数据
  #参数estimator为基模型
  #参数n_features_to_select为选择的特征个数
  rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2, verbose=3)
  rfe.fit(iris.data, iris.target)
  rfe.ranking_
  ```
  RFECV会通过交叉验证选出最佳的特征数量[4]：

  ```python
  from sklearn.feature_selection import RFECV
  all_features = [...]
  rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=12, n_jobs=-1)
  rfecv = RFECV(estimator=rfr, step=10, 
                cv=KFold(y.shape[0], n_folds=5, shuffle=False, random_state=101),
                scoring='neg_mean_absolute_error', verbose=2)
  rfecv.fit(X, y)
  sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
  
  print(' Optimal number of features: %d' % rfecv.n_features_)
  print(' The selected features are {}'.format(sel_features))
  
  # Save sorted feature rankings
  ranking = pd.DataFrame({'Features': all_features})
  ranking['Rank'] = np.asarray(rfecv.ranking_)
  ranking.sort_values('Rank', inplace=True)
  ```

- Stability selection:

  The high level idea is to apply a feature selection algorithm on different subsets of data and with different subsets of features. After repeating the process a number of times, the selection results can be aggregated, for example by checking how many times a feature ended up being selected as important when it was in an inspected feature subset. [9]

  Inject more noise into the original problem by generating bootstrap samples of the data, and to use a base feature selection algorithm (like the LASSO) to find out which features are important in every sampled version of the data. The results on each bootstrap sample are then aggregated to compute a *stability score* for each feature in the data. Features can then be selected by choosing an appropriate threshold for the stability scores. [10]

  N：有放回采样（bootstrap sampling）的次数；

  $$\lambda$$：模型正则化参数，a grid of values；

  k：第k个特征；

  $$\hat{\Pi}^{\lambda}_k$$：当正则化参数为$$\lambda$$时，特征k被选择的概率；

  $$\hat{S}^{\lambda}_i​$$：第$$i​$$次采样，模型正则化参数为$$\lambda​$$时选出来的特征子集；

  $$\hat{S}^{stable}\subset\{1,\dots,p\}​$$：最终输出的特征选择子集，其中$$p​$$为总特征个数.

  

  **step1:采样**， 对于每个正则化参数$$\lambda$$:

  1. for i=1,...,N:

     对原样本$$X^{n*p}$$进行有放回采样，采样的大小为$$\frac{n}{2}$$;

     在采样的样本上训练模型（如lasso）选择出特征的子集$$\hat{S}^{\lambda}_i$$,

  2. 计算在$$\lambda$$固定时，不同的采样样本选出特征k的概率$$\hat{\Pi}^\lambda_k=\mathbb{P}[k\in\hat{S}^\lambda]=\frac{1}{N}\sum_{i = 1}^N\mathbb{I}(k\in\hat{S}_i^\lambda)$$

  **step2:评分**， $$\hat{S}^{stable} = \{k: \max\limits_{\lambda}\hat{\Pi}_k^{\lambda}\ge\pi_{thres}\}$$，其中$$\pi_{thres}$$为一个预设的阈值

  使用[10]的stability-selction库，只要模型中带有`coef_ `或者`feature_importances`属性均可。

## Embedded特征选择

- Linear Model + 正则化：[7]

  该方法只适用于所有的特征都经过同样的scale，这样重要的特征对应的系数大，不重要的特征系数接近于0.这个方法适合简单的线性问题，并且数据的噪声不大。但如果问题的特征有multicollinearity：有多个线性相关的特征，将导致线性模型不稳定，即数据有微小的变化会导致模型的系数有较大的改变。例如有一个模型理论上为$$Y=X_1+X_2$$，但观测上有误差$$\hat{Y}=X_1+X_2+\epsilon$$，假设X1和X2是线性相关的$$X_1\approx X_2$$，误差会导致模型训练之后可能为$$Y=2X_1$$（只与X1有关）或者$$Y=-X_1+3X_2$$（X2是强正相关，X1是负相关），但事实上X1和X2都是等价值的正相关。

  加入正则项可以修正。L1正则项具有稀疏解的特性，适合特征选择。但L1同样不稳定，multicollinearity带来的问题依然存在。另外，L1没选到的特征不代表不重要，因为两个高相关性的特征可能只保留了一个。如果要确定哪个特征重要，再通过L2正则交叉验证。L2的效果不同于L1，L2会使特征的系数均分，用L2训练的模型更加稳定。尽管L2不如L1适合做特征选择，L2更适合做特征的解释。

  ```python
  lr = LinearRegression(normalize=True)
  lr.fit(X,Y)
  print(np.abs(lr.coef_)) # 模型不稳定，数据的噪声对模型的系数影响较大
  
  ridge = Ridge(alpha = 7)
  ridge.fit(X,Y)
  print(np.abs(ridge.coef_) # 模型同样不稳定，但可以得到稀疏解；无法确定选出来的特征是否重要
  
  lasso = Lasso(alpha=.05)
  lasso.fit(X, Y)
  print(lasso.coef_) # 验证Lasso选择的特征，检验是否其他共线性的特征表现更好；模型稳定
  ```

- Random Forest: [8]

  Mean decrease impurity: 分类问题使用Gini impurity 或者 information gain/entropy，回归问题用variance。在sklearn中使用`rf.feature_importances_`直接得到。**Mean decrease impurity的缺点**：1.存在bias，倾向于选择取值多的特征。2.如果存在多个correlated特征，从模型角度上出发，其中任何一个都可以用来做预测，没有明显偏好；但是一旦其中一个被使用，其他的correlated特征的重要性将大大降低。如果我们的目的是降低过拟合，那这个问题不重要；但如果我们想要解释模型，这个问题会误导我们，只有一个特征是重要的，其他与之correlated的特征都不重要，然而事实上他们与label的关系是相似的。

  Mean decrease accuracy: 直接衡量每个特征对模型accuracy的影响，将每个特征的顺序打乱permutate然后查看模型accuracy下降了多少。如果特征不重要，accuracy下降少；特征重要accuracy下降多。**此方法是在模型训练完之后，比较测试数据在某特征permutate前后metrics的变化程度。（模型只需要训练一次）**

  ```python
  from sklearn.cross_validation import ShuffleSplit
  from sklearn.metrics import r2_score
   
  X = boston["data"]
  Y = boston["target"]
   
  rf = RandomForestRegressor()
  scores = {}
   
  #crossvalidate the scores on a number of different random splits of the data
  for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
      X_train, X_test = X[train_idx], X[test_idx]
      Y_train, Y_test = Y[train_idx], Y[test_idx]
      rf.fit(X_train, Y_train)
      acc = r2_score(Y_test, rf.predict(X_test))
      for i in range(X.shape[1]):
          X_t = X_test.copy()
          np.random.shuffle(X_t[:, i])
          shuff_acc = r2_score(Y_test, rf.predict(X_t))
          scores[names[i]].append((acc-shuff_acc)/acc)
  print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
  # output: [(0.7276, 'LSTAT'), (0.5675, 'RM'), (0.0867, 'DIS'), (0.0407, 'NOX'), (0.0351, 'CRIM'), (0.0233, 'PTRATIO'), (0.0168, 'TAX'), (0.0122, 'AGE'), (0.005, 'B'), (0.0048, 'INDUS'), (0.0043, 'RAD'), (0.0004, 'ZN'), (0.0001, 'CHAS')]
  # LSAT and RM are 2 features that strongly impact model performance.
  ```


- Boruta [11]

  通常意义上用机器学习进行特征选择的目标是: 筛选出可以**使得当前模型cost function最小**的特征集合。如果删除某个特征，模型性能不变，未必说明该特征不相关，只能说明对于减小模型cost function没有帮助。Boruta的目标是：选出与label相关的特征，而不是选出另cost function最小的特征集合。Boruta能保留对模型有显著贡献的**所有**特征，这与很多特征降维方法使用的“最小最优特征集”思路相反。

  `Boruta` is produced as an improvement over `random forest` variable importance. Boruta在回归和分类问题上均可使用，要求基模型带有`feature_impotances_`属性。

  1. duplicate dataset, shuffle the values in each columns, to obtain **shadow features**
  2. train a classifier on merged dataset (real + shadow), and get importances of features. (树的ensemble模型的优势：对非线形的复杂数据支持；适合处理小样本多特征的数据。尽管这些模型也会过拟合，但是比其他模型过拟合速度慢)
  3. check for each real feature if they have higher importance than the best of shadow features. If they do, record this in a vector.
  4. At every iteration, check if a given feature is doing better than random chance, by simply comparing the number of times a feature did better than the shadow features using a binomial distribution. ![compare real feature with the max of shadow features](http://danielhomola.com/wp-content/uploads/2015/05/boruta2.png) 例如：F1在3次迭代的结果都优于shadow feature的最大值，使用binomial分布（k=3,n=3,p=0.5）计算p-value，如果p-value小于0.01就认为F1是与lable有相关性的，并删除F1对其余特征继续迭代。如果某特征连续15次没有超过shadow feature的max值，直接否决该特征，删除。持续这样做，直到达到迭代次数（或者所有特征都入选或否决）。

## 参考资料

sklearn.feature_selection模块适用于样本的特征选择/维数降低

1. [特征选择](https://zhuanlan.zhihu.com/p/32749489)

2. [Statistical meaning of pearsonr() output in Python](https://stats.stackexchange.com/questions/64676/statistical-meaning-of-pearsonr-output-in-python)

3. (kaggle)[Feature Ranking RFE, Random Forest, linear models](https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-models)

   Compare different kinds of feature ranking: Stability selection, recursive feature elimination, linear model, random forest feature ranking. Then create a feature ranking matrix, each column presents one feature ranking, using the function `ranking` to scale the ranking from 0 to 1. 

   最后，seaborn的pariplot(feature distribution)、heatmap(feature correlation)和factorplot(catplot)很漂亮。


4. [Recursive feature elimination](https://www.kaggle.com/tilii7/recursive-feature-elimination/code)

5. [Boruta feature elimination](https://www.kaggle.com/tilii7/boruta-feature-elimination)

6. [Feature selection – Part I: univariate selection](https://blog.datadive.net/selecting-good-features-part-i-univariate-selection/) **精华**

    Univariate selection examines each feature individually to determine the strength of the relationship of the feature with the lable

7. [Selecting good features – Part II: linear models and regularization](http://blog.datadive.net/selecting-good-features-part-ii-linear-models-and-regularization/)
   Lasso produces sparse solutions and as such is very useful selecting a strong subset of features for improving model performance. Ridge regression on the other hand can be used for data interpretation due to its stability and the fact that useful features tend to have non-zero coefficients.

8. [Selecting good features – Part III: random forests](http://blog.datadive.net/selecting-good-features-part-iii-random-forests/)

9. [Selecting good features – Part IV: stability selection, RFE and everything side by side](https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/)

10. https://thuijskens.github.io/2018/07/25/stability-selection/
     https://thuijskens.github.io/2017/10/07/feature-selection/

11. [BorutaPy – an all relevant feature selection method](http://danielhomola.com/2015/05/08/borutapy-an-all-relevant-feature-selection-method/)
