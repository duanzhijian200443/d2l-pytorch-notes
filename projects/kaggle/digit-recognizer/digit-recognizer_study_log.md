# Kaggle Digit Recognizer 学习日志 (Study Log)

## 📌 当前学习进度 (Current Progress)
- **最近更新时间**: 2026-09-17
- **当前项目**: Kaggle `Digit Recognizer`
- **当前阶段**: Digit Recognizer 实战基本收尾。已从原始 LeNet Baseline 逐步完成 ReLU、MaxPool、通道扩展、BatchNorm、Adam、weight decay、最佳 checkpoint、学习率搜索、scheduler / 数据增强实验，并完成 42,000 张带标签数据的全量训练与最终提交。
- **第一次 Kaggle Public Score**: **0.92800**
- **最高 Kaggle Public Score**: **0.99085**（全量 42,000 训练后再次提交仍为 `0.99085`）
- **当前本地最高 val acc**: **0.9902380952**（约 `99.02%`，4,200 张验证集）
- **本地 Baseline 最终结果**:
  - `train loss ≈ 0.2516`
  - `train acc ≈ 0.9213`
  - `val acc ≈ 0.9340`
- **当前结论**: 项目已完成从 `0.92800` Baseline 到 `0.99085` Kaggle 成绩的完整优化闭环。当前主要收益来自更合适的 CNN 结构、ReLU/MaxPool、BatchNorm、Adam、checkpoint 与学习率选择；scheduler、数据增强和 42,000 全量重训在本轮实验中没有继续提高 Kaggle 分数。

### 核心掌握概念
- Kaggle Digit Recognizer 的 `train.csv` Shape 为 `(42000, 785)`：1 列 `label` + 784 个像素；`test.csv` Shape 为 `(28000, 784)`，没有标签。
- 训练集、验证集、测试集的职责严格区分：
  - **train**：参与反向传播和参数更新；
  - **validation**：不参与参数更新，用于观察泛化能力、调超参数和选模型；
  - **test**：Kaggle 隐藏真实标签，只用于最终推理和提交，不能直接计算本地 accuracy。
- 从 `train.csv` 中拆出验证集是为了避免用 Kaggle test 反复调参造成测试集信息泄露。当前采用 9:1：
  - `train_dataset = 37800`
  - `val_dataset = 4200`
- `random_split(..., generator=torch.Generator().manual_seed(42))` 只固定数据拆分结果；模型权重初始化、DataLoader shuffle 等仍有自己的随机源。
- 若要让实验更可复现，应在**模型创建/权重初始化之前**调用 `torch.manual_seed(42)`。固定 seed 的本质是固定随机过程，不是让模型天然变强；不能不断试 seed 后只挑最好的一次作为模型能力。
- Pandas → PyTorch 的数据链：
  - `DataFrame/Series`
  - `to_numpy()`
  - `torch.tensor(...)`
  - `TensorDataset`
  - `random_split`
  - `DataLoader`
- 图像输入从 `(N, 784)` reshape 为 `(N, 1, 28, 28)`，其中 `N` 是样本数，`1` 是灰度通道，`28×28` 是空间尺寸。
- 像素值原始范围约为 `0~255`，通过 `X / 255.0` 归一化到 `[0,1]`。
- 输入图片使用 `torch.float32`；分类标签使用 `torch.long`（即 `torch.int64`），以满足 `nn.CrossEntropyLoss()` 的类别索引要求。
- `TensorDataset(X, y)` 将 `X[i]` 与 `y[i]` 一一配对；`DataLoader` 再负责 batch、shuffle 和迭代。
- `train_iter` 使用 `shuffle=True`；`val_iter` 和 `test_iter` 使用 `shuffle=False`。尤其测试集不能打乱，否则 prediction 顺序会和 Kaggle `ImageId` 错位。
- 训练统计不再依赖 `d2l.Accumulator`：
  - `loss_sum`：累计样本 loss 总和；
  - `correct`：累计预测正确样本数；
  - `total`：累计样本总数；
  - `train_loss = loss_sum / total`
  - `train_acc = correct / total`
- `CrossEntropyLoss()` 默认给出当前 batch 的平均 loss，因此累计整个 epoch 时使用 `loss_sum += l.item() * X.shape[0]`。
- 分类预测使用 `y_hat.argmax(dim=1)`，从每个样本 10 个 logits 中取最大值对应的类别索引。
- 验证/测试阶段应调用 `net.eval()`，并使用 `torch.inference_mode()` 关闭训练所需的 Autograd 跟踪。
- Kaggle test 没有 `y`，因此测试循环只做 `X → GPU → net(X) → argmax(dim=1) → 收集预测`。
- GPU 上的预测 Tensor 通过 `.cpu()` 搬回 CPU，再通过 `.tolist()` 转成普通 Python list。
- `predictions.extend(pred.cpu().tolist())` 会把每个 batch 的预测展开追加到同一个一维列表；最终已验证 `len(predictions) == 28000`。
- `sample_submission.csv` 是 Kaggle 提供的提交格式模板。最终只需替换 `Label` 列，再通过 `to_csv(..., index=False)` 生成正式提交文件。
- 10 分类随机猜测时，交叉熵约为 `log(10)≈2.3026`、accuracy 约为 `10%`。训练初期若长期出现 `loss≈2.30`、`acc≈0.10`，说明模型基本仍处于接近随机猜测状态。
- 当前经典 LeNet 使用 Sigmoid + SGD(`lr=0.9`) 对随机初始化较敏感，曾观察到不同初始化下收敛速度和最终验证准确率明显不同；固定主要随机源后实验更便于比较。
- 当前 Baseline 的本地验证 `0.9340` 与 Kaggle Public Score `0.92800` 差约 0.6 个百分点，说明验证集与 Kaggle 测试表现大体一致。

---

## 🧪 Baseline 实验记录

### Experiment-001 — 原始 LeNet Baseline
- **数据集**: Kaggle Digit Recognizer
- **训练样本**: 37,800
- **验证样本**: 4,200
- **测试样本**: 28,000
- **Split seed**: `42`
- **训练随机种子**: 在模型初始化前固定 `torch.manual_seed(42)`
- **Batch size**: `256`
- **输入 Shape**: `(N, 1, 28, 28)`
- **输入归一化**: `/255.0`
- **模型结构**:
  - `Conv2d(1, 6, kernel_size=5, padding=2)`
  - `Sigmoid`
  - `AvgPool2d(kernel_size=2, stride=2)`
  - `Conv2d(6, 16, kernel_size=5)`
  - `Sigmoid`
  - `AvgPool2d(kernel_size=2, stride=2)`
  - `Flatten`
  - `Linear(16*5*5, 120)`
  - `Sigmoid`
  - `Linear(120, 84)`
  - `Sigmoid`
  - `Linear(84, 10)`
- **权重初始化**: `Xavier uniform`
- **Loss**: `nn.CrossEntropyLoss()`
- **Optimizer**: `torch.optim.SGD`
- **Learning rate**: `0.9`
- **Epochs**: `10`
- **最终 train loss**: `≈0.2516`
- **最终 train acc**: `≈0.9213`
- **最终 val acc**: `≈0.9340`
- **Kaggle Public Score**: **0.92800**
- **定位**: 后续所有改动先与本 Baseline 比较，一次尽量只改一个变量，避免无法判断提升来源。

### 后续候选实验
- [x] `Sigmoid → ReLU`
- [x] 调整 SGD / Adam 学习率并比较收敛稳定性
- [x] `SGD → Adam`
- [x] `AvgPool → MaxPool`
- [x] 调整卷积通道数（`6/16 → 32/64`）
- [x] 增加训练轮数并观察 train/val gap
- [x] 数据增强（`RandomRotation`，本轮未带来明确收益）
- [x] 保存最佳验证模型，而不是只使用最后一个 epoch
- [ ] 多个 seed 重复实验，报告平均值与波动

---


## 🧪 进阶实验记录（2026-09-16 ～ 2026-09-17）

### Experiment-002 — `Sigmoid → ReLU` 与随机性控制
- 将 LeNet 中的 `Sigmoid` 改为 `ReLU` 后，前几轮收敛速度显著加快；同时也观察到在学习率过大、随机过程未完全控制时，训练可能长期停留在 `loss≈2.3026 / acc≈10%` 的随机猜测状态。
- 明确区分两类 seed：
  - `torch.manual_seed(42)`：控制 PyTorch 全局随机序列，影响模型初始化等 Torch 随机操作；
  - `torch.Generator().manual_seed(42)`：创建一个独立 RNG，可只交给 `random_split` / `DataLoader` 等指定操作。
- seed 不会“保存参数”；它只是让随机序列从同一状态开始。真正保存训练后参数需要 `torch.save(...)`。
- 当前策略：固定 train/val split；模型初始化 seed 用于公平比较，不靠反复换 seed 挑最好结果。

### Experiment-003 — 保存最佳验证模型（Best Checkpoint）
- 新增：
  ```python
  best_acc = 0
  if val_acc > best_acc:
      best_acc = val_acc
      torch.save(net.state_dict(), "best.pt")
  ```
- 训练结束后的内存中 `net` 默认是**最后一个 epoch** 的参数，不一定是验证集最好的参数。
- 测试 / Kaggle 提交前显式恢复：
  ```python
  net.load_state_dict(torch.load("best.pt"))
  net.eval()
  ```
- `state_dict()` 保存模型权重和 bias 等参数；不自动保存网络结构、optimizer 状态或 epoch。
- 选择 checkpoint 看 `val_acc` 而不是 `train_acc`：训练集参与参数更新，验证集更适合衡量泛化与模型选择。

### Experiment-004 — 从经典 LeNet 扩展为更宽 CNN
- 主要结构改动：
  - `Conv 1→6→16` → `Conv 1→32→64`
  - `AvgPool2d` → `MaxPool2d`
  - `Sigmoid` → `ReLU`
  - 加入 `BatchNorm2d(32/64)`
- 当前卷积 Shape 重新推导：
  - `(1,28,28)` → Conv(p=2,k=5) → `(32,28,28)`
  - MaxPool(2) → `(32,14,14)`
  - Conv(k=5,p=0) → `(64,10,10)`
  - MaxPool(2) → `(64,5,5)`
  - Flatten → `64*5*5=1600`
- 因此全连接入口应为 `nn.Linear(64*5*5, 128)`，不能沿用不同结构下的 `64*4*4`。
- `BatchNorm2d(C)` 中的 `C` 是卷积输出通道数，不是 batch size；训练时使用 batch 统计量，`eval()` 时使用运行统计量。
- 本阶段本地验证最好曾提升到约 `0.977`，随后继续优化进入 `0.98+`。

### Experiment-005 — `SGD → Adam`、weight decay 与学习率搜索
- 优化器切换到 `torch.optim.Adam` 后，合理学习率集中在 `1e-3` 附近，而不是早期 SGD 使用的 `0.2 / 0.9` 数量级。
- 典型对照：
  - `lr=0.001`：收敛快，本地最好约 `0.987`；
  - `lr=0.0005`：更平缓，但本次没有超过 `0.001`；
  - 继续在 `≈5.5e-4` 附近细调，最终本地最高 `val_acc = 0.9902380952380953`。
- 当前一次高分配置记录：
  ```python
  optimizer = torch.optim.Adam(
      net.parameters(),
      lr≈0.0005509,
      weight_decay=0.01
  )
  ```
- weight decay 的效果不是固定的：在早期 SGD / 结构下，`0.001`、`0.01` 曾导致表现下降或波动；在后续 Adam + BN + 更宽 CNN 的组合里，`0.01` 又能与高分配置共存。结论是必须把它视为与 optimizer / lr / 模型结构联动的超参数，而不是“越小/越大越好”。

### Experiment-006 — Scheduler 对照：`CosineAnnealingLR`
- 尝试：
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=20
  )
  ```
- 本次实验最高验证准确率约 `0.98595`，低于固定学习率已经取得的 `0.99+`。
- 结论：scheduler 不是必然提升；如果当前固定学习率已经合适，过早/过强地衰减学习率可能让后半程更新幅度过小。
- 最终提交配置不强行堆 scheduler，以验证结果为准。

### Experiment-007 — 数据增强：`RandomRotation`
- 学习了自定义 `Dataset` 的 `__len__ / __getitem__`，用于在每次取样时动态执行图像变换。
- `RandomRotation(10)` 属于**数据管线**，不是网络层；训练时每个 epoch 可能看到不同旋转版本。
- 加入旋转增强后，训练耗时明显增加：增强通常在 CPU / DataLoader 侧执行，小型 MNIST CNN 很快，GPU 容易等待 CPU 数据预处理，因此数据增强反而成为瓶颈。
- 本轮实验没有证明旋转增强优于当前最佳固定数据方案，因此没有作为最终配置。

### Experiment-008 — 本地验证与 Kaggle Score 不必严格一致
- 观察到两类现象：
  - 某次本地 `val_acc > 0.99`，Kaggle 仅 `0.98+`；
  - 另一次本地未突破 `0.99`，Kaggle 却达到 `0.99085`。
- 原因：本地验证集只有 `4200` 张，是从 42,000 训练数据中抽出的有限样本；Kaggle 测试数据是另一批未见样本。两者难度与样本构成不同，单次 split 的局部排名不能保证和 Kaggle 排名完全一致。
- 因此验证集应主要用于**模型选择和趋势比较**，而不是把 `val_acc` 当成 Kaggle 分数的精确预测。

### Experiment-009 — 42,000 全量训练与最终提交
- 调参阶段使用：`37,800 train + 4,200 validation`。
- 最终实验移除验证集，将 `train.csv` 的全部 **42,000** 张带标签图像用于训练，再对 28,000 张 `test.csv` 推理。
- 全量训练后再次提交 Kaggle：**`0.99085`**，与此前最高提交相同，没有继续提升。
- 结论：多 4,200 张训练样本并不保证排行榜立刻提高；最终效果仍受训练轮数、初始化、超参数、测试样本边界案例等因素影响。

### 当前 Kaggle 成绩轨迹
| 阶段 | Kaggle Score | 备注 |
|:---|---:|:---|
| Baseline | `0.92800` | 原始 LeNet / Sigmoid / SGD |
| 中间改进 | `0.96810` | CNN 结构与训练策略开始优化 |
| 进一步优化 | `0.98903` | 更宽 CNN + BN / Adam 等组合 |
| 当前最高 | **`0.99085`** | 学习率/结构等继续调优 |
| 42,000 全量重训 | **`0.99085`** | 分数与当前最高持平 |

### 暂缓到正式 Kaggle 竞赛再学习
- K 折交叉验证（K-Fold）：用于更稳定地评估模型，并可训练多个 fold 模型。
- 模型融合（Ensemble）：多个 seed / fold / 不同结构模型对测试概率求平均或投票。
- 当前项目不继续堆复杂度，保留为以后正式比赛阶段的学习内容。

## ✅ 项目 Checklist
- [x] 下载并读取 `train.csv` / `test.csv`
- [x] 识别训练集与测试集 Shape 差异
- [x] 拆分 `label` 与像素特征
- [x] Pandas → NumPy → Tensor
- [x] `X` 使用 `float32`，`y` 使用 `long`
- [x] `(N,784) → (N,1,28,28)`
- [x] 像素 `/255.0` 归一化
- [x] `TensorDataset(X,y)`
- [x] `random_split` 拆分 train / validation
- [x] 固定 split seed
- [x] `DataLoader` 构造 train / val / test iterator
- [x] 自己实现 `train_one_epoch`
- [x] 去除训练统计对 `d2l.Accumulator` / `d2l.accuracy` 的依赖
- [x] 自己实现验证 accuracy
- [x] 使用 `torch.inference_mode()` 做验证/推理
- [x] GPU 训练和推理
- [x] 收集 28,000 个测试预测
- [x] 生成 `submission.csv`
- [x] 第一次 Kaggle 提交
- [x] Public Score = `0.92800`
- [x] 中间提交：`0.96810` / `0.98903`
- [x] 当前最高 Kaggle Score = `0.99085`
- [x] `BatchNorm2d` 实验
- [x] `Adam` + 学习率细调
- [x] `best.pt` checkpoint 保存/加载
- [x] `CosineAnnealingLR` 对照实验（未提升）
- [x] 42,000 全量训练并最终提交（仍为 `0.99085`）
- [x] 完成多轮 Baseline 对照实验并达到 Kaggle `0.99085`

---

## 🗂️ 异常与问题工单

> **格式：V6.0 四步极简法则**
>
> ### 🐛 [Bug-XXX] 问题摘要
> - **💻 触发代码**: `代码`
> - **🩸 原始报错 / 现象**: 系统报错或观察到的异常
> - **🧠 底层原因**: 白话解释；涉及 Tensor 时标明 Shape / dtype / device
> - **💊 修复方案**: 正确代码或处理策略
>
> *纯理论问题可省略原始报错，保留“核心疑问 / 底层解释 / 结论”。*

### 🐛 [Bug-001] `enumerate(train_iter)` 解包方式错误
- **💻 触发代码**: `for (X, y) in enumerate(train_iter):`
- **🩸 原始现象**: 循环变量结构与预期 `(X,y)` 不一致。
- **🧠 底层原因**: `train_iter` 本身每次产生 `(X,y)`；`enumerate(train_iter)` 产生的是 `(i,(X,y))`。
- **💊 修复方案**:
  - 不需要 batch 索引：`for X, y in train_iter:`
  - 需要 batch 索引：`for i, (X, y) in enumerate(train_iter):`

### 🐛 [Bug-002] epoch 统计变量放在 batch 循环内部导致反复清零
- **💻 触发代码**:
  ```python
  for X, y in train_iter:
      loss_sum = 0.0
      correct = 0
      total = 0
  ```
- **🩸 原始现象**: 训练指标只反映当前/最后一个 batch，而不是整个 epoch。
- **🧠 底层原因**: 每进入一个 batch，累计器就重新归零，历史 batch 统计全部丢失。
- **💊 修复方案**: 将 `loss_sum/correct/total` 放到 batch 循环之前，每个 epoch 只初始化一次。

### 🐛 [Bug-003] loss 累计对象写错
- **💻 触发代码**: `loss_sum += loss_fn * X.shape[0]`
- **🩸 原始现象**: 把损失函数对象本身当成数值参与累计。
- **🧠 底层原因**: `loss_fn` 是 `nn.CrossEntropyLoss()` 模块；真正的当前 batch loss 是 `l = loss_fn(y_hat, y)`。
- **💊 修复方案**: `loss_sum += l.item() * X.shape[0]`

### 🐛 [Bug-004] 验证函数中 `nn.Module` 类型名写错
- **💻 触发代码**: `isinstance(net, torch.nn.module)`
- **🩸 原始报错**: `AttributeError: module 'torch.nn' has no attribute 'module'`
- **🧠 底层原因**: PyTorch 模块基类名称为 `nn.Module`，`Module` 的 `M` 必须大写。
- **💊 修复方案**: `isinstance(net, nn.Module)`

### 🐛 [Bug-005] `.numel()` 调用到了 accuracy 返回的 float 上
- **🩸 原始报错**: `AttributeError: 'float' object has no attribute 'numel'`
- **🧠 底层原因**: accuracy 统计值已经是 Python 数值，`.numel()` 应作用于 Tensor `y` 而不是 float。
- **💊 修复方案**:
  ```python
  pred = y_hat.argmax(dim=1)
  correct += (pred == y).sum().item()
  total += y.numel()
  ```

### ❓ [工单-006] 为什么不用 Kaggle `test.csv` 拆验证集？
- **核心疑问**: 是否可以从测试集拿一部分当 validation。
- **🧠 底层解释**: Kaggle `test.csv` 没有真实 `label`，本地无法计算 accuracy；更重要的是，test 应尽量保持为最终评估数据，不能在开发过程中反复用于模型选择。
- **结论**: 从带标签的 `train.csv` 拆 train/validation，Kaggle `test.csv` 仅做最终预测。

### 🐛 [Bug-007] `test_iter` 未定义 / 错把测试集送入 accuracy 函数
- **💻 触发代码**: `test_acc = evaluate_accuracy_gpu(net, test_iter)`
- **🩸 原始报错**: `NameError: name 'test_iter' is not defined`
- **🧠 底层原因**: 当时只构造了训练/验证 DataLoader；而 Kaggle test 无标签，即使定义 `test_iter` 也不能用现有 accuracy 函数评估。
- **💊 修复方案**:
  - 训练期间：`val_acc = evaluate_accuracy_gpu(net, val_iter, device)`
  - 训练结束：单独构造 `test_iter`，只做 forward + prediction。

### 🐛 [Bug-008] `DataLoader` 构造测试集时漏传 dataset
- **💻 触发代码**:
  ```python
  test_iter = DataLoader(
      batch_size=256,
      shuffle=False
  )
  ```
- **🩸 原始现象**: DataLoader 不知道要从哪里读取样本。
- **🧠 底层原因**: `DataLoader` 的第一个核心输入必须是 dataset / Tensor。
- **💊 修复方案**:
  ```python
  test_iter = DataLoader(
      X_test,
      batch_size=256,
      shuffle=False
  )
  ```

### 🐛 [Bug-009] Pandas 对象被覆盖成 Tensor 后继续调用 `.to_numpy()`
- **💻 触发代码**:
  ```python
  y = torch.tensor(y.to_numpy(), dtype=torch.long)
  X_test = torch.tensor(y.to_numpy(), dtype=torch.float32)
  ```
- **🩸 原始报错**: `AttributeError: 'Tensor' object has no attribute 'to_numpy'`
- **🧠 底层原因**: 第一行执行后 `y` 已经是 Tensor；`.to_numpy()` 属于 Pandas/NumPy 转换链，不属于 PyTorch Tensor。
- **💊 修复方案**:
  ```python
  X_test = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
  ```
  并推荐用 `y_series` / `x_df` 等变量名避免类型覆盖。

### 🐛 [Bug-010] Jupyter Cell 报错后变量处于“半更新”状态
- **🩸 原始现象**: 代码已经改对，但重新运行后行为仍显得异常，甚至 Cell 长时间忙碌。
- **🧠 底层原因**: Jupyter 是有状态内核；一个 Cell 中前几行成功执行后，即使后面报错，已经修改过的变量不会自动回滚。
- **💊 修复方案**:
  1. 先 `Interrupt` 当前执行；
  2. 测试简单 `print("hello")`；
  3. 若内核仍异常，再 `Restart Kernel`；
  4. 从上到下重新运行必要 Cell。

### 🐛 [Bug-011] 同一个 `d2l` 环境中一个 Notebook 能看到 GPU、另一个看不到
- **🩸 原始现象**:
  - Notebook A：`torch.cuda.is_available() == True`
  - Notebook B：`False`
  - 两者 `sys.executable` 与 PyTorch CUDA 版本相同。
- **🧠 底层原因**: 不同 Notebook 使用不同的 Jupyter kernel 进程；某个 kernel 的 CUDA 初始化状态异常时，不代表 PyTorch 安装或显卡整体失效。
- **💊 修复方案**: 重启异常 Notebook 的 kernel。重启后 CUDA 恢复，模型重新连接 RTX 5060 Laptop GPU。

### ❓ [工单-012] 为什么固定随机种子后准确率变化很大？
- **核心疑问**: 同一网络重启后曾出现约 `0.77 / 0.88 / 0.93` 等明显不同的验证表现。
- **🧠 底层解释**:
  - Xavier 权重初始化是随机的；
  - `DataLoader(shuffle=True)` 的 batch 顺序也是随机的；
  - 经典 `Sigmoid + SGD(lr=0.9)` LeNet 对初始状态较敏感；
  - `random_split` 的 seed 只固定数据划分，并不会自动固定模型初始化。
- **结论**: 在模型创建/初始化前固定 `torch.manual_seed(42)`，用于实验可复现和公平比较；不能靠不断试 seed 挑最高分来宣称模型更强。

### ❓ [工单-013] 为什么前几轮几乎不学习，后面突然起飞？
- **核心疑问**: 训练初期 `loss≈2.30`、accuracy≈10%，随后某个 epoch accuracy 快速上升。
- **🧠 底层解释**: 10 分类随机预测的交叉熵理论值约为 `log(10)=2.3026`；早期模型几乎没有形成有效区分。随着卷积特征与分类层参数逐步进入更有效区域，预测开始偏离均匀分布，准确率会出现非线性加速。Sigmoid 的饱和/梯度特性也会放大这种“前期慢、后期突然学起来”的现象。
- **结论**: 后续可用 ReLU、不同优化器/学习率进行对照。

### ❓ [工单-014] `.cpu()`、`.tolist()` 与 `extend()` 在提交预测中的作用
- **核心疑问**: 为什么测试预测要写 `predictions.extend(pred.cpu().tolist())`？
- **🧠 底层解释**:
  - `.cpu()`：把 CUDA Tensor 搬回 CPU；
  - `.tolist()`：把 Tensor 转成 Python list；
  - `.extend()`：把一个 batch 中的多个预测值展开追加到总列表，而不是形成嵌套 list。
- **结论**: 最终得到长度为 `28000` 的一维 predictions，与 Kaggle test 行顺序一一对应。

### 🐛 [Bug-015] 测试集不能 `shuffle=True`
- **核心问题**: 测试 DataLoader 是否也可以打乱。
- **🧠 底层原因**: Kaggle submission 的第 1 行预测必须对应 `ImageId=1`，第 2 行对应 `ImageId=2`……如果测试数据被 shuffle，prediction 顺序会与模板 ImageId 错位。
- **💊 修复方案**: `test_iter = DataLoader(X_test, batch_size=256, shuffle=False)`

### ❓ [工单-016] 为什么 `index=False` 必须保留？
- **核心疑问**: `submission.to_csv("submission.csv", index=False)` 中 `index=False` 的作用。
- **🧠 底层解释**: Pandas DataFrame 自带 `0,1,2,...` 行索引；若一起写入 CSV，会多出 Kaggle 不需要的额外列。
- **结论**: 提交文件仅保留 `ImageId,Label` 两列。

---


### ❓ [工单-017] `torch.manual_seed` 和独立 `Generator` 有什么区别？
- **核心疑问**: 开头已经 `torch.manual_seed(42)`，`random_split(..., generator=torch.Generator().manual_seed(42))` 是否重复？
- **🧠 底层解释**: 前者设置 PyTorch 全局 RNG；后者新建一个局部 RNG，并只影响显式接收该 generator 的操作。两者可以使用不同 seed，例如模型初始化用 `42`、数据划分用 `41`，结果就是模型初始随机序列与数据划分各自固定但不同。
- **结论**: 不是参数存档，而是随机数流控制。学习阶段可保留独立 split generator，方便保证验证集不变。

### ❓ [工单-018] 为什么保存 `best.pt` 而不是直接用训练结束后的 `net`？
- **核心疑问**: 最后一个 epoch 不是已经训练得最多了吗？
- **🧠 底层解释**: SGD/Adam 的训练指标并不单调；后续 epoch 可能让 train acc 继续提高，但 val acc 下降。训练循环结束后的 `net` 是最后一轮参数，而 `best.pt` 可以保留历史上验证集最优时刻。
- **结论**: 提交前先 `load_state_dict(best.pt)`，否则可能误用最后一轮退化模型。

### ❓ [工单-019] 为什么模型结构改了以后 `Linear` 输入维度必须重算？
- **核心疑问**: `64*4*4` 与 `64*5*5` 到底由什么决定？
- **🧠 底层解释**: Linear 输入来自最后一层卷积/池化输出的 `C×H×W`。当前网络的 Shape 为 `28→28→14→10→5`，因此是 `64*5*5`。
- **结论**: 改 kernel / padding / stride / pooling 后必须重新推导 H、W，不能照搬旧网络常数。

### ❓ [工单-020] 为什么 `weight_decay` 加上后有时更差，有时又能出高分？
- **核心疑问**: `0.001` / `0.01` 的效果不稳定。
- **🧠 底层解释**: weight decay 是与模型容量、optimizer、学习率、训练轮数联动的正则超参数；它不是独立的“性能开关”。过强会欠拟合，合适时可抑制参数过度增长。
- **结论**: 对照实验决定是否保留，不能只凭经验固定一个数。

### ❓ [工单-021] 为什么加入 `RandomRotation` 后训练明显变慢？
- **核心疑问**: GPU 训练模型没变，为什么 wall-clock 时间明显增加？
- **🧠 底层解释**: 随机旋转属于输入数据预处理，通常在 CPU/DataLoader 侧逐样本执行。MNIST CNN 本身很小，GPU 很快，CPU augmentation 很容易成为新的瓶颈。
- **结论**: 数据增强有计算成本；是否值得取决于泛化收益，本轮未作为最终配置。

### ❓ [工单-022] 为什么本地 validation 和 Kaggle Score 会“反着来”？
- **核心疑问**: 有时本地超过 0.99、Kaggle 只有 0.98+；有时本地没到 0.99、Kaggle 却到 0.99085。
- **🧠 底层解释**: 两者不是同一批样本。4,200 张本地验证集只是一份有限抽样，包含的难例比例会影响 accuracy；Kaggle 测试集是另一批未见样本。因此模型在两个集合上的相对表现可以不同。
- **结论**: 用 validation 选模型、看趋势；不要要求它精确预测排行榜分数。

### ❓ [工单-023] 为什么 42,000 全量重训后 Kaggle 分数没有变？
- **核心疑问**: 训练样本从 37,800 增至 42,000，理论上信息更多，但分数仍为 `0.99085`。
- **🧠 底层解释**: 更多数据通常有利，但排行榜 accuracy 是离散指标；新增数据可能只改变部分低置信样本，而未改变被计分样本的最终 argmax，或者收益被训练轮数/超参数差异抵消。
- **结论**: 全量训练是合理最终流程，但不是“必涨分按钮”。

## 🛠️ API 速查表 (API Cheatsheet)
> 当前覆盖 Digit Recognizer Baseline 端到端数据、训练、验证、推理与提交链路。

### 📄 Pandas / CSV
| 当你想要... | 用这个 | 核心作用 |
|:---|:---|:---|
| 读取 CSV | `pd.read_csv("train.csv")` | 得到 Pandas `DataFrame` |
| 读取某一列 | `df["label"]` | 得到标签 `Series` |
| 删除标签列保留特征 | `df.drop(columns=["label"])` | 得到纯像素特征 |
| DataFrame/Series → NumPy | `x.to_numpy()` | 去掉 Pandas 标签结构，得到数值数组 |
| 查看前几行 | `df.head()` / `df.head(10)` | 快速检查格式 |
| 写出 Kaggle CSV | `df.to_csv("submission.csv", index=False)` | 不额外保存 Pandas 行索引 |

### 📦 Dataset / DataLoader
| 当你想要... | 用这个 | 核心作用 / Shape |
|:---|:---|:---|
| 配对输入和标签 | `TensorDataset(X, y)` | `dataset[i] -> (X[i], y[i])` |
| 随机拆分数据集 | `random_split(dataset, [37800,4200], generator=...)` | 生成 train / val 子集 |
| 固定全局 Torch RNG | `torch.manual_seed(seed)` | 固定模型初始化等 PyTorch 随机序列 |
| 构造随机生成器 | `torch.Generator().manual_seed(42)` | 固定特定随机操作的序列 |
| 构造 mini-batch | `DataLoader(dataset, batch_size=256, shuffle=True)` | 批量读取训练数据 |
| 取一个 batch | `next(iter(train_iter))` | 常用于检查 `X/y` Shape |

### 🔢 Tensor / dtype / Shape
| 当你想要... | 用这个 | 核心作用 |
|:---|:---|:---|
| NumPy → Tensor | `torch.tensor(a, dtype=...)` | 创建 PyTorch Tensor |
| 图像 dtype | `torch.float32` | 网络输入/卷积计算常用 |
| 标签 dtype | `torch.long` | 等价 `torch.int64`；分类标签常用 |
| 恢复图像 Shape | `X.reshape(-1,1,28,28)` | `(N,784) -> (N,1,28,28)` |
| 像素归一化 | `X / 255.0` | `0~255 -> 0~1` |
| 样本数/标签元素数 | `y.numel()` | 返回元素总数 |

### 🧠 训练 / 验证
| 当你想要... | 用这个 | 核心作用 |
|:---|:---|:---|
| 训练模式 | `net.train()` | 启用训练行为 |
| 评估模式 | `net.eval()` | 切换为稳定评估/推理行为 |
| 清梯度 | `optimizer.zero_grad()` | 防止 batch 间梯度累积 |
| 前向传播 | `y_hat = net(X)` | 得到 logits |
| 交叉熵 | `nn.CrossEntropyLoss()` | logits + long label |
| BatchNorm（CNN） | `nn.BatchNorm2d(C)` | 对卷积输出的每个通道做批归一化，`C`=通道数 |
| 最大池化 | `nn.MaxPool2d(2)` | 保留局部最大响应并下采样 |
| 反向传播 | `l.backward()` | 计算参数梯度 |
| 更新参数 | `optimizer.step()` | 按优化器规则更新权重 |
| Adam 优化器 | `torch.optim.Adam(net.parameters(), lr=...)` | 自适应每个参数的更新步长 |
| 保存模型参数 | `torch.save(net.state_dict(), "best.pt")` | 保存当前权重 / bias 等参数 |
| 恢复模型参数 | `net.load_state_dict(torch.load("best.pt"))` | 恢复已保存的参数到同结构网络 |
| 余弦学习率调度 | `CosineAnnealingLR(optimizer, T_max=...)` | 按余弦曲线逐步降低 lr；需配合 `scheduler.step()` |
| 预测类别 | `y_hat.argmax(dim=1)` | `(N,10) -> (N,)` |
| 正确样本数 | `(pred == y).sum().item()` | Tensor 比较后求和转 Python 标量 |
| 纯推理模式 | `with torch.inference_mode():` | 比 `no_grad` 更明确地面向验证/部署推理 |

### 🖥️ Device / GPU
| 当你想要... | 用这个 | 核心作用 |
|:---|:---|:---|
| 自动选设备 | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | CUDA 可用时选择 GPU |
| 模型迁移 | `net.to(device)` | 模型参数迁移到同一设备 |
| batch 迁移 | `X.to(device)` / `y.to(device)` | 输入与标签必须和模型同设备 |
| 检查 CUDA | `torch.cuda.is_available()` | CUDA 是否可用 |
| 查看 GPU 数量 | `torch.cuda.device_count()` | 当前 kernel 能看到几个 CUDA device |
| 查看 GPU 名称 | `torch.cuda.get_device_name(0)` | 查看第 0 张 GPU |
| GPU Tensor → CPU | `x.cpu()` | 便于后续 Python/Pandas 处理 |


### 🖼️ 数据增强 / Dataset
| 当你想要... | 用这个 | 核心作用 |
|:---|:---|:---|
| 随机旋转训练图片 | `transforms.RandomRotation(10)` | 每次取样时随机旋转一定角度 |
| 自定义数据集 | `class MyDataset(Dataset)` | 在 `__getitem__` 中动态应用 transform |
| 返回数据集长度 | `__len__()` | 告诉 DataLoader 样本数 |
| 定义单样本读取 | `__getitem__(index)` | 返回 `(img, label)`，可在此执行增强 |

### 📤 Kaggle 推理 / Submission
| 当你想要... | 用这个 | 核心作用 |
|:---|:---|:---|
| Tensor → Python list | `x.tolist()` | 把预测 Tensor 转普通列表 |
| 展开追加 batch 预测 | `predictions.extend(batch_preds)` | 保持最终 predictions 为一维 list |
| 检查预测数量 | `len(predictions)` | 本比赛应为 `28000` |
| 读取提交模板 | `pd.read_csv("sample_submission.csv")` | 保留 Kaggle 要求的 `ImageId` 格式 |
| 写入预测 | `submission["Label"] = predictions` | 替换模板标签列 |
| 保存提交文件 | `submission.to_csv("submission.csv", index=False)` | 生成 Kaggle 可提交 CSV |

---


## 🧭 项目阶段结论 / 后续

Digit Recognizer 第一轮系统实践到此基本收尾：

1. **结果**：Kaggle 从 `0.92800` 提升到 **`0.99085`**；42,000 全量重训后再次提交仍为 `0.99085`。
2. **已掌握**：CSV→Tensor→DataLoader、CNN Shape、ReLU / MaxPool、BatchNorm、SGD / Adam、weight decay、seed、checkpoint、validation model selection、GPU inference、submission。
3. **已做对照但未证明有收益**：`CosineAnnealingLR`、`RandomRotation`、单纯扩大训练轮数、全量重训。
4. **暂缓**：K-Fold、Ensemble、TTA 等比赛技巧，等正式参加 Kaggle 竞赛时再系统学习。
5. **D2L 主线**：实践结束后可返回卷积神经网络主线，下一阶段继续学习更现代的 CNN 架构，并把本项目作为后续网络结构实验的轻量测试台。
