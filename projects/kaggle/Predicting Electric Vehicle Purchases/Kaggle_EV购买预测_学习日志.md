# Kaggle 电动车购买预测学习日志


> 项目：Kaggle Playground Series S6E9 — Predicting Electric Vehicle Purchases  
> 学习时间：2026-09-30 ～ 2026-10-01  
> 当前状态：阶段性暂停，回到深度学习主线  
> 目标：通过一次真实表格二分类比赛，补齐树模型、AUC、CatBoost、特征工程与 Kaggle 提交流程的基本认知。


## 1. 数据与任务


- 训练集：668,665 行 × 15 列。
- 测试集：286,571 行 × 14 列。
- 标签列：`Will_Buy_EV`。
- 标签分布：
  - No：约 82.5355%
  - Yes：约 17.4645%
- 去掉 `id` 与标签列后，共 13 个输入特征。
- 主要类别特征：
  - `Gender`
  - `City_Type`
  - `Current_Car_Type`
  - `Home_Charging_Possible`
  - `Subsidy_Available`
  - `Range_Anxiety_Level`


### 数据划分


使用：


```python
from sklearn.model_selection import train_test_split


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```


理解：
- 与 PyTorch 的 `random_split` 目的相同，都是从训练数据中划分训练集和验证集。
- `stratify=y` 用于保持训练集与验证集中的正负样本比例一致。
- Kaggle 的 `test.csv` 相当于官方隐藏测试集；本地 `validation` 用于调参与模型选择。


## 2. 决策树与 GBDT 核心理解


### 决策树


决策树中的规则不是手工写死的，而是模型从数据中学习得到的，包括：
- 选择哪个特征进行切分；
- 选择什么阈值或类别；
- 先判断哪个条件、后判断哪个条件。


例如树可能学到类似：


```text
Annual_Income_USD > 某阈值？
├─ 是 -> 继续判断其他特征
└─ 否 -> 进入另一分支
```


`depth` 控制一棵树最多可以学习多少层规则。深度越大，模型容量越强，但过拟合风险也越高。


### Gradient Boosting


核心更新形式：


```text
F_(t+1)(x) = F_t(x) + η f_t(x)
```


理解：
- `F_t(x)`：前面已经训练好的所有树的组合。
- `f_t(x)`：新加入的一棵树，用来修正旧模型的错误。
- `η`：learning rate / shrinkage，控制每棵新树的修正力度。


平方误差下：


```text
残差 = y - F_t(x)
     = - ∂L / ∂F
```


因此“拟合残差”可以看成“拟合负梯度”的特殊情况。


重要认知：
- GBDT 是加法模型，但不是关于输入 x 的线性模型。
- 多棵树做线性加权组合，并不意味着整体对 x 是线性的；单棵树本身就是分段非线性函数。


## 3. CatBoost Baseline


使用的基础参数：


```python
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    eval_metric="AUC",
    cat_features=cat_cols,
    verbose=100
)


model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100
)
```


### 参数理解


- `iterations=1000`：最多添加 1000 棵树，不等同于神经网络 epoch。
- `learning_rate=0.05`：控制每棵新树对整体模型的修正幅度。
- `depth=8`：每棵树的最大深度。
- `eval_metric="AUC"`：训练时重点监控验证集 AUC。
- `cat_features=cat_cols`：告诉 CatBoost 哪些输入列是类别特征。
- `verbose=100`：每 100 轮打印一次日志。
- `eval_set=(X_valid, y_valid)`：指定验证集。
- `early_stopping_rounds=100`：若连续 100 轮没有刷新历史最佳验证 AUC，则提前停止。


### 训练日志理解


例如：


```text
900: test: 0.9413454 best: 0.9413595 (842)
```


含义：
- `test`：此处指 `eval_set` 的验证 AUC，不是 Kaggle 的 test.csv。
- `best`：从第 0 轮到当前轮为止的历史最佳验证 AUC。
- `(842)`：最佳结果出现在 iteration 842。
- `total`：当前已经消耗的训练时间。
- `remaining`：按当前速度估算的剩余训练时间。


最终 baseline：
- `best_iteration = 842`
- 本地验证 AUC：约 **0.94135948**
- Kaggle Public Leaderboard：约 **0.94108**


本地验证分数与 Public LB 很接近，说明当前验证划分与提交流程总体可靠。


## 4. ROC-AUC 的理解


AUC 不是准确率。


Accuracy 关注：
- 用一个阈值把概率转成 0/1 后，预测正确了多少样本。


AUC 更关注：
- 正样本的预测分数是否通常高于负样本。
- 可以直观理解为：随机抽一个正样本和一个负样本，模型把正样本排在前面的概率。


由于本比赛标签不平衡（Yes 约 17.46%），AUC 比单纯 Accuracy 更合适。


验证集预测：


```python
pred = model.predict_proba(X_valid)[:, 1]
score = roc_auc_score(y_valid, pred)
```


理解：
- `predict_proba(X_valid)` 返回每个样本属于各类别的概率。
- 二分类时第 0 列是 No 概率，第 1 列是 Yes 概率。
- `[:, 1]` 取每个样本为正类 Yes 的概率，用于计算 AUC。


## 5. Kaggle 提交流程


测试集不包含真实标签，因此本地不能计算 test AUC。


流程：


```python
X_test = test.drop(columns=["id"])
test_pred = model.predict_proba(X_test)[:, 1]


submission = sample.copy()
submission["Will_Buy_EV"] = test_pred
submission.to_csv("submission.csv", index=False)
```


注意：
- `sample_submission.csv` 只是提交格式模板。
- 模板里的默认概率（例如 0.174645）不是模型预测结果。
- 必须把 `test_pred` 写入 `Will_Buy_EV` 后再保存与提交。
- Kaggle 最终提交的是预测结果 CSV，而不是模型文件。


## 6. Feature Importance


CatBoost 训练后查看：


```python
importance = model.get_feature_importance()


sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
)
```


本次较重要的特征大致为：


- `Subsidy_Available`：约 49.83
- `Environmental_Concern_Level`：约 18.36
- `Annual_Income_USD`：约 10.43
- `Age`：约 3.98
- `Daily_Commute_km`：约 3.83


理解：
- Feature Importance 是模型训练后计算出的贡献度，不是数据集自带字段。
- 数值越高，通常表示模型越依赖该特征。
- Feature Importance 不等于因果关系，也不等于“删除后 AUC 会按同样百分比下降”。


## 7. 消融实验：删除特征


Baseline：


```text
AUC ≈ 0.94135948
```


### 删除 Daily_Commute_km


- 验证 AUC：约 **0.94058130**
- best iteration：约 649
- 相对 baseline：下降约 **0.00078**


结论：
- 有一定贡献，但影响较小。


### 删除 Age


- 验证 AUC：约 **0.94100730**
- best iteration：约 810
- 相对 baseline：下降约 **0.00035**


结论：
- Age 有独立贡献，但不是核心决定因素。


### 删除 Subsidy_Available


- 验证 AUC：约 **0.88949204**
- best iteration：约 942
- 相对 baseline：下降约 **0.05187**


结论：
- `Subsidy_Available` 是极强预测特征。
- 删除后模型只能依赖其他弱特征慢慢补偿，因此需要更多 boosting rounds，且整体性能明显下降。


## 8. 特征工程实验


特征工程的基本思想：


```text
原始特征 X -> 人工构造新表示 φ(X) -> 重新训练 -> 比较验证 AUC
```


关键原则：
- 新特征通常基于旧特征构造。
- 特征越多不一定越好。
- 每次最好只改变一个思路，用验证集判断是否真的有效。


### 实验 A：Income_per_Age


构造：


```python
X["Income_per_Age"] = (
    X["Annual_Income_USD"] / X["Age"]
)
```


结果：
- 验证 AUC：约 **0.94131380**
- best iteration：约 895
- 相对 baseline：基本无提升，略降。


结论：
- CatBoost 已能从 Income 与 Age 的树分裂组合中捕捉大部分关系。
- 人工增加比值没有带来明显的新信息。


### 实验 B：Charging_Score + Home_Charging


根据家庭充电条件与附近充电站信息构造新特征。


结果：
- 验证 AUC：约 **0.94127036**
- best iteration：约 782
- 相对 baseline：基本无提升，略降。


结论：
- 新特征可能让模型更早学到已有关系，因此 best iteration 下降；
- 但没有增加新的泛化信息，因此最终 AUC 没有提升。
- “训练更快”不等于“泛化更好”。


### 实验 C：类别交叉组合


类别交叉示例：


```python
X["City_Car_Type"] = (
    X["City_Type"].astype(str)
    + "_"
    + X["Current_Car_Type"].astype(str)
)
```


如果 A 有 m 个类别、B 有 n 个类别，则理论组合空间最多为 m × n（笛卡尔积），实际出现的组合可能更少。


结果：
- 验证 AUC：约 **0.94127496**
- 相对 baseline：没有提升，略降。


结论：
- CatBoost 本身已经擅长类别特征与类别交互；
- 简单人工交叉更多是在重复已有信息，而没有提供真正新的信号。


## 9. 当前实验结论


目前可以较确定地认为：


1. CatBoost 已经能较充分地利用当前 13 个原始特征。
2. 简单数学组合（Income/Age）没有提升。
3. 简单充电条件组合没有提升。
4. 简单类别交叉没有提升。
5. Feature Importance 与消融实验总体一致：重要性越高的特征，删除后通常影响越大。
6. 目前性能瓶颈并不在“缺少几个显而易见的人工特征”。


## 10. 工程问题与排错记录


### sklearn 安装后仍无法 import


现象：


```text
ModuleNotFoundError: No module named 'sklearn'
```


原因：
- VS Code Notebook 使用的是 Conda 环境：
  `C:\Users\34331\miniconda3\envs\d2l\python.exe`
- 普通 `!pip install scikit-learn` 安装到了系统 Python，环境不一致。


解决：


```python
import sys
!{sys.executable} -m pip install scikit-learn
```


### y 中出现大量 NaN


现象：


```text
ValueError: Input y contains NaN.
```


检查发现 `y.isna().sum()` 大量非零，而原始 `Will_Buy_EV` 只有 `No` / `Yes`。


处理：
- 检查并重新建立标签映射：


```python
y = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})
```


### X_valid 未定义


原因：
- 变量曾误写成 `X_vaild` / `y_vaild`。


经验：
- 表格实验中变量名较多，拼写错误可能直到训练结束才暴露。
- 后续统一使用 `X_train, X_valid, y_train, y_valid`。


### catboost_info


CatBoost 默认生成 `catboost_info/`，其中保存训练日志，例如：
- `learn_error.tsv`
- `test_error.tsv`
- `time_left.tsv`
- `catboost_training.json`


这些是训练记录，不是模型本身，也不是 ChatGPT 生成的文件。


## 11. 对高分 Notebook 的初步认识


阅读了一个高分方案 notebook 后发现，其核心已不是“单独训练一个 CatBoost”，而是更高级的 Kaggle 工程：


- 多模型预测结果融合；
- 利用模型间相关性寻找 diversity；
- 优化 blend 权重；
- Fisher / probit 等融合方法；
- 规则后处理；
- 使用多个已有高分 submission 进行 ensemble。


重要认知：
- 顶分阶段的关键资源往往是“模型多样性”，而不仅是某一个模型单独更强。
- 目前阶段直接照抄这类方案学习成本过高，会偏离深度学习主线。


## 12. 阶段性决定


本项目暂时停在这里。


已经完成的学习闭环：


```text
读取数据
-> 数据检查
-> 训练/验证划分
-> CatBoost baseline
-> AUC 理解
-> early stopping
-> feature importance
-> 消融实验
-> 简单特征工程
-> Kaggle submission
-> Public Leaderboard 验证
```


后续如果重新参加表格赛，再继续补：


- LightGBM
- XGBoost
- Stratified K-Fold / OOF
- 更系统的 EDA
- 超参数搜索
- Target Encoding
- 模型融合 / Ensemble
- Rank / Fisher / weighted blending


当前优先级恢复为深度学习主线，避免继续追榜挤占 D2L、序列模型、Attention、Transformer 与 mini-GPT 的学习时间。