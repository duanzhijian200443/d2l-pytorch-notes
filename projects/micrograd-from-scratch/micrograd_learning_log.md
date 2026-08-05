# Micrograd 学习日志

## 📌 当前进度

- **最近更新时间**：2026-08-05
- **项目名称**：Micrograd From Scratch
- **当前阶段**：已完成 `Neuron`、`Layer` 与基础 `MLP` 的独立构造，已打通多层前向数据流与多输出计算图可视化；下一步进入损失函数、参数收集与训练循环
- **当前里程碑**：M6（已完成）— 已实现 `Neuron`、`Layer`、`MLP`，掌握全连接结构、层间维度传递、局部变量重绑定、多输出梯度路径与完整计算图合并；准备进入 M7
- **代码状态**：当前本地 Notebook 中的 `Value` 引擎、`Neuron`、`Layer` 与 `MLP` 均已可运行；已用单一汇总根节点绘制整个多输出网络的计算图。本次仅合并更新日志，未上传或覆盖 Google Drive 文件

## 🎯 项目目标

通过纯 Python 从零实现一个标量级自动微分引擎，并进一步搭建简单神经网络，重点理解：

- 计算图如何在前向传播时动态建立
- 链式法则如何被程序化为反向传播
- 梯度为什么需要按拓扑顺序回传
- 神经元、层和 MLP 如何建立在自动微分引擎之上
- PyTorch 自动微分机制背后的核心设计

## ✅ 已完成内容

### 1. `Value` 类基础结构

已理解并实现以下核心属性：

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
```

- `data`：保存当前节点的标量数值
- `_prev`：保存当前节点的直接前驱节点
- `_op`：记录生成当前节点的运算类型
- `_children=()`：默认没有前驱节点，表示叶子节点
- `set(_children)`：将前驱节点保存为集合，便于计算图遍历和去重

### 2. 对象显示

已实现并理解：

```python
def __repr__(self):
    return f"Value={self.data}"
```

`__repr__` 只控制对象在调试和交互环境中的显示方式，不改变对象本身的数据或计算行为。

### 3. 运算符重载

已实现：

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    return out

def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    return out
```

已理解：

- `__add__` 与 `__mul__` 类似 C++ 的操作符重载
- `a + b` 会调用 `a.__add__(b)`
- `a * b` 会调用 `a.__mul__(b)`
- `self` 指向左操作数对象
- `other` 指向右操作数对象
- `(self, other)` 是包含两个 `Value` 对象的元组，作为新节点的直接前驱
- 新运算不会修改原节点，而是返回一个新的 `Value` 节点

### 4. 前向计算图构建

已完成并验证类似计算：

```python
a = Value(-2.0)
b = Value(3.0)
c = Value(10.0)
d = a * b + c
```

计算结果：

```text
d.data = 4.0
d._op = '+'
```

即使没有显式写出中间变量：

```python
e = a * b
d = e + c
```

乘法仍会创建中间节点。该节点被 `d._prev` 间接保留，因此计算图不会丢失。

### 5. 计算图可视化与 Graphviz

已完成 Graphviz 环境配置，并理解视频中的：

```python
trace(root)
draw_dot(root)
```

二者属于辅助工具，不属于自动微分引擎核心：

- `trace`：沿 `_prev` 递归收集节点和边
- `draw_dot`：使用 `graphviz.Digraph` 渲染计算图
- `draw_dot` 不是 Graphviz 自带 API，而是作者在 Notebook 中定义的辅助函数
- `label` 用于图中显示节点名，`grad` 用于显示当前节点梯度

### 6. 节点梯度与默认反向函数

已理解 `Value` 节点后续需要增加：

```python
self.grad = 0.0
self._backward = lambda: None
```

- `grad` 保存最终输出对当前节点的导数
- `lambda: None` 是默认空操作，使叶子节点也能统一调用 `_backward()`
- 运算节点会用自己的局部求导闭包覆盖这个默认函数

### 7. 加法与乘法的局部 `_backward`

已理解并写出局部反向传播结构：

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out


def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

关键点：

- 嵌套 `_backward` 形成闭包，记住本次运算的 `self`、`other` 和 `out`
- 前向阶段只保存求导规则，不立即执行
- `out._backward = _backward` 保存函数对象；不能写成 `_backward()`
- 梯度必须使用 `+=`，因为同一节点可能沿多条路径影响最终结果
- 加法中的 `1.0 * out.grad` 可以简写为 `out.grad`；显式写 `1.0` 只是强调局部导数为 1

### 8. `out.grad` 与上游梯度

对于：

```python
out = self * other
```

`out.grad` 表示：

\[
\frac{\partial L}{\partial out}
\]

它不是当前节点主动取得的，而是由后续节点执行其 `_backward()` 时累加写入。

例如：

```python
c = a * b
L = c * 4
```

反向顺序为：

```text
L.grad = 1
→ L._backward() 写入 c.grad = 4
→ c._backward() 读取 c.grad，再写入 a.grad 与 b.grad
```

因此：

```text
下游节点的 _backward：写入当前节点的 grad
当前节点的 _backward：读取自己的 grad，乘局部导数后传给前驱
_prev：用于后续拓扑排序，保证调用顺序正确
```

当前已经理解：`_prev` 会在统一 `backward()` 中用于递归访问直接前驱并构造拓扑序。

### 9. `tanh` 的前向与局部反向传播

已理解以下结构：

```python
def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out
```

关键结论：

- `t = tanh(x)` 是前向阶段保存的中间数值
- `1 - t**2` 是局部导数 \(\frac{\partial out}{\partial self}\)
- `out.grad` 是上游梯度 \(\frac{\partial L}{\partial out}\)
- 整行代码对应链式法则：

\[
\frac{\partial L}{\partial self}
\mathrel{+}=
\frac{\partial L}{\partial out}
\frac{\partial out}{\partial self}
=
out.grad\,(1-t^2)
\]

### 10. 闭包的作用

闭包可以概括为：

```text
函数本身 + 函数定义时依赖的外层变量环境
```

Micrograd 中的嵌套 `_backward()` 会记住本次前向运算对应的：

- 输入节点 `self`
- 另一个输入节点 `other`（二元运算）
- 前向中间值，如 `t`
- 输出节点 `out`

因此外层运算函数返回后，节点仍能在反向阶段调用自己的局部求导规则。不同运算生成的闭包代码可以相同，但捕获的具体节点不同。

### 11. 拓扑排序与统一 `backward()`

已理解 Micrograd 使用 DFS 后序遍历建立拓扑顺序：

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0

    for node in reversed(topo):
        node._backward()
```

各部分职责：

- `v._prev`：找到当前节点的直接前驱
- `visited`：保证共享节点在整张图中只进入拓扑列表一次
- `topo.append(v)` 放在递归之后：先加入前驱，再加入当前节点，形成合法前向拓扑序
- `self.grad = 1.0`：因为最终输出对自身的导数为 \(\frac{\partial L}{\partial L}=1\)
- `reversed(topo)`：从最终输出向叶子节点执行局部 `_backward()`

必须区分：

```text
backward()   ：整张图的调度器，只从最终节点调用一次
_backward()  ：单个运算节点保存的局部求导规则
```

### 12. 分支、共享节点与梯度累积

已通过练习理解：

- `_prev` 只保存直接前驱，不保存全部祖先
- `set(_children)` 只能在当前节点的一组直接前驱内去重；整张图的去重依赖 `visited`
- 同一节点可以通过多条路径影响最终结果，梯度遵循“各路径贡献相加”
- 对 `b = a * a`，`b._prev` 中只有一个不同对象 `a`，但乘法闭包中的 `self` 和 `other` 都指向 `a`，因此两份局部梯度都必须累加
- 对 `L = a * a + a`：

\[
\frac{\partial L}{\partial a}=2a+1
\]

使用 `=` 会覆盖已有路径的贡献；使用 `+=` 才能得到正确结果。


### 13. 普通常数包装与反向运算符

已实现并理解普通数字与 `Value` 的兼容机制：

```python
other = other if isinstance(other, Value) else Value(other)
```

由此可让普通 `int/float` 参与计算图运算。已理解：

- `isinstance(other, Value)` 负责类型判断
- 条件为假时将普通数字包装成常数 `Value`
- `assert` 用于拒绝非法输入，不负责类型转换
- `3 * a` 会先尝试左操作数的方法；失败后由 Python 调用 `a.__rmul__(3)`
- 实例方法中的 `self` 始终绑定到实际调用该方法的对象

已实现：

```python
def __rmul__(self, other):
    return self * other
```

### 14. 扩展基础算子

已实现并验证：

```python
def __sub__(self, other):
    return self + other * (-1)

def __pow__(self, other):
    ...

def __truediv__(self, other):
    return self * other**(-1)

def exp(self):
    ...
```

核心理解：

- 减法复用加法和乘法
- 除法复用乘法和幂运算
- `a / b` 被改写为 `a * b**(-1)`
- Python 中 `**` 的优先级高于 `*`
- `__pow__` 当前只支持常数指数 `int/float`
- 所有局部反向规则必须使用 `+=` 累积梯度

### 15. 用基础算子组合实现 `tanh`

已不再为 `tanh` 单独手写局部导数，而是使用已有算子组合：

```python
def tanh(self):
    t = (self * 2).exp()
    return (t - 1) / (t + 1)
```

对应公式：

\[
\tanh(x)=\frac{e^{2x}-1}{e^{2x}+1}
\]

该实现会自动构建由乘法、指数、加减法、幂和除法组成的计算图，反向传播由各基础节点沿拓扑逆序自动完成。

已验证：

```text
x = 2
tanh(x) ≈ 0.96402758
x.grad ≈ 0.07065082
```

其中：

\[
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
\]



### 16. 构造单个神经元 `Neuron`

已理解并实现：

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
```

结构含义：

- `nin` 表示该神经元接收的输入数量
- `self.w` 保存 `nin` 个相互独立的 `Value` 权重
- `self.b` 保存一个 `Value` 偏置
- `__call__` 执行一次前向传播
- `act` 对应仿射变换：

\[
act=\sum_{i=1}^{nin}w_i x_i+b
\]

- `out` 对应激活输出：

\[
out=\tanh(act)
\]

已掌握的 Python 机制：

```text
Neuron(3) → 创建对象并执行 __init__
n(x)      → 对已创建对象执行 __call__
```

`n(x)` 近似等价于：

```python
n.__call__(x)
# 或
Neuron.__call__(n, x)
```

类中即使还存在 `parameters()`、`forward()` 等其他普通方法，`n(x)` 也只会触发 Python 为“对象调用语法”固定绑定的 `__call__`。普通方法必须显式按名称调用。

#### 神经元初始化与前向传播中的关键结论

- 权重和偏置必须包装为 `Value`，而不是普通 `float`，否则无法保存计算图、梯度与局部反向规则
- 当前项目中的权重是自定义 `Value`，不是 PyTorch `Tensor`
- 随机初始化的主要作用是打破神经元之间的对称性，不是直接防止过拟合
- `sum(generator, self.b)` 将偏置作为累计初值，因此偏置只加入一次
- `zip(self.w, x)` 只配对，不负责相乘；两侧长度不同时会按较短一侧静默截断
- 输入可以参与求导，但“能获得梯度”不等于“应作为模型参数更新”；正常训练只更新 `w` 和 `b`

已完成手算：

\[
\frac{\partial out}{\partial act}=1-out^2
\]

\[
\frac{\partial out}{\partial w_i}=(1-out^2)x_i
\]

\[
\frac{\partial out}{\partial b}=1-out^2
\]

### 17. 构造全连接层 `Layer`

已独立实现并确认当前代码正确：

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs
```

结构含义：

- `nin`：每个神经元接收的输入数量
- `nout`：该层包含的神经元数量，也是该层输出数量
- `Layer(3, 4)` 创建 4 个相互独立的 `Neuron(3)`
- 同一组输入 `x` 会分别传给该层的每个神经元
- 每个神经元使用自己的独立权重和偏置计算一个输出
- 当前实现始终返回列表，因此 `Layer(3, 1)` 的输出仍是 `[Value(...)]`，而不是直接的 `Value`

一层的参数总数为：

\[
nout\times(nin+1)
\]

例如：

\[
Layer(5,3):\quad 3\times(5+1)=18
\]

#### 对象独立性

正确写法：

```python
[Neuron(nin) for _ in range(nout)]
```

每次循环都会创建一个新的神经元对象。

错误写法：

```python
[Neuron(nin)] * nout
```

该写法虽然得到长度为 `nout` 的列表，但列表中的所有位置都引用同一个 `Neuron` 对象。修改任意位置的参数都会影响所有位置，多个“神经元”也会产生完全相同的输出。

#### 梯度流向

属于同一个 `Layer`，不代表所有参数在每次反向传播中都必然获得梯度。参数是否被反向访问，取决于它是否位于当前损失节点可达的计算图中。

例如：

```python
outs = [o1, o2, o3]
loss = o1 + o2 + o3
```

三个神经元的参数都参与损失，都会接收到梯度。

若：

```python
loss = outs[0]
```

则只有第一个神经元的参数位于当前 `loss` 的反向路径上；另外两个神经元的参数梯度通常保持为 `0.0`。


### 18. 构造多层感知机 `MLP`

已独立实现并运行：

```python
class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

结构含义：

- `nin`：整个网络的输入维度
- `nouts`：每一层的输出维度列表
- `sizes = [nin] + nouts`：把输入维度与各层输出维度放入同一列表
- 每一对相邻尺寸 `sizes[i] → sizes[i+1]` 对应一个 `Layer`

例如：

```python
MLP(3, [4, 4, 1])
```

会构造：

```text
sizes = [3, 4, 4, 1]
Layer(3, 4)
Layer(4, 4)
Layer(4, 1)
```

即：

```text
3 维输入 → 4 个隐藏单元 → 4 个隐藏单元 → 1 个输出单元
```

`range(len(sizes) - 1)` 的边界来自相邻区间数量：`sizes` 有 4 个维度位置，但只有 3 个连接区间。也可等价写为 `range(len(nouts))`。

#### `MLP.__call__` 中 `x` 的重绑定

调用：

```python
x = [2.0, 3.0]
model = MLP(2, [3, 4, 5])
out = model(x)
```

进入 `model.__call__(x)` 时：

```text
外部变量 x ─┐
            ├──→ 同一个原始输入列表
局部参数 x ─┘
```

执行：

```python
x = layer(x)
```

不是修改原始列表，也不是让 `x` 指向 `Layer` 对象，而是让函数内部的局部变量 `x` 重新指向当前层返回的新列表。

数据流为：

```text
局部 x：长度 2 的 float 列表
→ Layer(2, 3)：长度 3 的 Value 列表
→ Layer(3, 4)：长度 4 的 Value 列表
→ Layer(4, 5)：长度 5 的 Value 列表
→ return x
```

外部变量 `x` 仍指向原始输入；外部变量 `out` 指向最后一层输出。若外部写成 `x = model(x)`，才会在函数返回后把外部 `x` 重新绑定到最终输出。

### 19. 全连接层的完整连接语义

若上一层输出为：

```text
[h1, h2, h3]
```

下一层是：

```python
Layer(3, 2)
```

则该层包含两个相互独立的 `Neuron(3)`。每个神经元都接收完整的三维输入：

\[
o_1=\tanh(w_{11}h_1+w_{12}h_2+w_{13}h_3+b_1)
\]

\[
o_2=\tanh(w_{21}h_1+w_{22}h_2+w_{23}h_3+b_2)
\]

连接关系为：

```text
h1 ─┬→ o1
    └→ o2
h2 ─┬→ o1
    └→ o2
h3 ─┬→ o1
    └→ o2
```

关键点：

- 所有输出神经元共享同一组输入
- 不同神经元不共享权重和偏置
- “全连接”表示每个上一层输出都允许连接到每个下一层神经元
- 某条连接训练后权重可能接近 0，但结构上仍存在

若权重矩阵按“每个输出神经元一行”组织，则：

\[
W\in\mathbb{R}^{nout\times nin}
\]

### 20. `MLP` 参数数量

设：

\[
n_0=nin,\qquad[n_1,n_2,\ldots,n_L]=nouts
\]

第 \(i\) 层参数量为：

\[
n_i(n_{i-1}+1)=n_{i-1}n_i+n_i
\]

其中：

- \(n_{i-1}n_i\)：权重数量
- \(n_i\)：偏置数量

总参数量：

\[
\sum_{i=1}^{L} n_i(n_{i-1}+1)
\]

例如：

```python
MLP(3, [4, 4, 1])
```

参数量：

```text
Layer(3,4)：3×4 + 4 = 16
Layer(4,4)：4×4 + 4 = 20
Layer(4,1)：4×1 + 1 = 5
总计：41
```

批量大小或样本数量增加，只会增加前向数据量、计算量和中间激活数量，不会创建新的模型参数。同一套权重会被所有样本复用。

### 21. `tanh` 作为激活函数及当前设计边界

当前 `Neuron.__call__` 中：

```python
act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
out = act.tanh()
```

这里的 `tanh` 就是激活函数。神经元分为：

```text
仿射变换：act = w·x + b
→ 非线性激活：out = tanh(act)
```

激活函数的核心作用是引入非线性，而不是字面意义上“把隐藏层启动”。如果所有层都只有线性变换，多层组合仍可合并为一个线性变换，深度不会带来本质表达能力提升。

当前教学实现把 `tanh` 硬编码在 `Neuron` 中，因此隐藏层和输出层都会执行 `tanh`，最终输出被限制在：

\[
-1<out<1
\]

这适合 Micrograd 教学与有界输出任务，但不是通用工程设计。激活函数可以改为 ReLU、Sigmoid 或恒等映射，前提是 `Value` 实现相应前向操作和局部反向规则。

一般设计原则：

- 隐藏层通常使用非线性激活
- 输出层是否使用激活以及使用何种激活，由任务的输出空间和损失函数决定
- 任意实数回归常用线性输出
- 二分类概率常用 Sigmoid
- 多分类通常结合 logits 与 Softmax / 交叉熵处理
- 目标位于 `[-1,1]` 时可考虑 `tanh`

更灵活的设计应让 `Neuron` 或 `Layer` 支持“是否激活”的配置，再由 `MLP` 决定隐藏层与输出层的设置；不能在当前 `Neuron` 已经执行 `tanh` 后，又在 `MLP.__call__` 中重复执行一次。

### 22. 多输出网络的完整计算图可视化

当前 `draw_dot(root)` 与 `trace(root)` 只从单个根节点向前回溯。因此：

```python
draw_dot(out[0])
```

只会绘制生成 `out[0]` 的计算图，不会自动包含其他并列输出。

已使用一个新的汇总根节点将所有输出合并：

```python
out = model(x)
O = out[0]

for n in out[1:]:
    O = O + n

O.backward()
draw_dot(O)
```

此时：

```text
out[0] ─┐
out[1] ─┼→ 连续加法形成 O → backward() / draw_dot(O)
out[2] ─┤
...     ─┘
```

因为每个输出都成为 `O` 的祖先节点，从单一根 `O` 回溯时会覆盖全部输出分支及其共享的前层计算图。

若：

\[
O=o_1+o_2+\cdots+o_k
\]

则：

\[
\frac{\partial O}{\partial o_i}=1
\]

因此图中各最终输出节点的 `grad` 显示为 `1.0000` 是正确结果。

必须区分：

```python
for n in out:
    n.backward()
```

会在同一组参数上连续执行多次反向传播并累积梯度；它不等价于先构造一个标量汇总根节点再统一 `backward()`。训练时通常应先定义一个标量损失节点，再调用一次 `loss.backward()`。


## 🧠 已掌握的关键理解

### `self.data` 为什么可以访问

`self` 和 `other` 在运行时指向实际的 `Value` 实例。每个实例在创建时都执行：

```python
self.data = data
```

因此：

```text
self.data  → 左侧 Value 对象的 data
other.data → 右侧 Value 对象的 data
```

参数名本身并不会赋予对象属性；能够访问 `.data`，是因为参数当前指向的对象拥有这个属性。

### `_prev` 的作用

`_prev` 保存当前节点的直接来源，例如：

```text
a ─┐
   ├─ * → temp ─┐
b ─┘            ├─ + → d
c ──────────────┘
```

对应：

```text
temp._prev = {a, b}
d._prev    = {temp, c}
```

它的主要用途是：

1. 从最终结果节点反向找回完整计算图
2. 确定梯度应传给哪些前驱节点
3. 为后续拓扑排序和 `backward()` 提供图结构

`_prev` 只保存直接前驱，不直接保存整个计算图；完整图由各节点递归连接形成。

### 参数与对象的关系

Python 中参数只是变量名，运行时指向传入对象。

```python
def __mul__(self, other):
```

执行 `a * b` 时可理解为：

```python
Value.__mul__(a, b)
```

因此：

```text
self  → a
other → b
```

`self` 不是关键字，而是 Python 实例方法第一个参数的标准命名惯例。


### 类调用与对象调用的分工

```text
类(...)     → 创建实例，并初始化实例
实例(...)   → 调用实例的 __call__
实例.方法() → 调用显式指定的普通方法
```

因此：

```python
n = Neuron(3)  # 创建神经元并初始化 w、b
out = n(x)     # 使用已有的 w、b 进行一次前向传播
```

同一个神经元只初始化一次，但可以使用同一套参数处理多组输入。

### `Layer` 的数据流与形状语义

```text
输入长度 nin
→ 同时送入 nout 个独立的 Neuron(nin)
→ 产生 nout 个 Value 输出
→ 当前实现以列表形式返回
```

`Layer` 只是组织多个神经元，并不会自动把多个输出求和。把 `outs` 再执行 `sum(outs)` 会把一层的向量输出错误压缩为单个标量。


## 🗂️ 已解决问题记录

### [Micrograd-001] `_children=()` 是什么

- 它是构造函数参数的默认值，不是内置构造函数
- `()` 表示空元组
- 叶子节点没有前驱时使用默认空元组

### [Micrograd-002] 为什么使用 `set(_children)`

- 将传入的前驱节点转换为集合
- 便于后续遍历和去重
- Micrograd 当前主要关心依赖关系，不依赖前驱顺序

### [Micrograd-003] `_op` 前导下划线是否必须

- 功能上不是必须
- 前导下划线是“内部使用”的命名惯例
- 建议统一使用 `_op`，便于与原实现和后续代码对照
- 默认值应为 `''`，不要写成包含空格的 `' '`

### [Micrograd-004] 一行表达式是否会丢失中间节点

不会。

```python
d = a * b + c
```

与：

```python
e = a * b
d = e + c
```

构建的计算图相同。区别仅在于显式变量 `e` 更方便调试。

### [Micrograd-005] `draw_dot` 为什么未定义

- `draw_dot` 不是 Graphviz 包内置函数
- 必须先运行作者提供的 `trace()` 与 `draw_dot()` 定义单元格
- Jupyter 重启内核后，函数定义会从内存中消失，需要重新按顺序执行

### [Micrograd-006] 可视化代码是否需要重点手写

- 不需要将 Graphviz API 当作 Micrograd 核心内容背诵
- 只需理解它沿 `_prev` 收集图结构并渲染
- 可视化主要用于检查计算图连接、运算符和梯度值

### [Micrograd-007] 为什么每个节点要保存 `_backward` 函数

- 前向传播时只能确定局部求导规则，尚不知道最终损失传来的上游梯度
- 将闭包保存到输出节点后，等反向阶段再执行
- 每次运算生成的闭包分别记住本次运算涉及的对象

### [Micrograd-008] `lambda: None` 的作用

- 它是一个无参数、无副作用的默认空函数
- 叶子节点没有前驱可继续传播，因此默认什么也不做
- 所有节点都拥有 `_backward` 后，反向遍历时无需额外判断属性是否存在

### [Micrograd-009] 加法中的 `1.0` 是否可以省略

可以。加法局部导数为 1：

```python
self.grad += 1.0 * out.grad
```

等价于：

```python
self.grad += out.grad
```

### [Micrograd-010] 乘法反向传播为什么必须使用 `+=`

- `=` 会覆盖此前通过其他路径累积的梯度
- `+=` 才符合多路径梯度相加规则
- 在 `a * a` 这类共享节点场景中，两个输入位置的梯度必须累加到同一个 `a.grad`

### [Micrograd-011] `out.grad` 如何获得上游梯度

- `out.grad` 是普通对象属性，不会主动寻找梯度
- 它由下游节点的 `_backward()` 写入
- 反向拓扑顺序保证当前节点执行时，上游梯度已经准备好

### [Micrograd-012] `_prev` 在统一 `backward()` 中如何使用

- `build_topo(v)` 通过 `for child in v._prev` 递归访问直接前驱
- `_prev` 提供图结构，`visited` 负责全图去重，二者职责不同
- 没有 `_prev`，无法从最终输出找回整张计算图；没有 `_backward`，找到节点后也不知道如何传递梯度

### [Micrograd-013] `_prev` 只保存直接前驱，不保存全部祖先

例如：

```python
c = a * b
d = c + a
L = d.tanh()
```

正确关系是：

```text
c._prev = {a, b}
d._prev = {c, a}
L._prev = {d}
```

完整计算图由这些局部关系递归连接形成。

### [Micrograd-014] `set(_children)` 与 `visited` 的去重范围不同

- `set(_children)`：只对当前节点的直接前驱去重
- `visited`：在 DFS 遍历整张计算图时去重
- 一个节点分别出现在多个节点的 `_prev` 中时，真正保证它只进入一次 `topo` 的是 `visited`

### [Micrograd-015] `topo.append(v)` 为什么必须放在递归之后

这是 DFS 后序遍历：

```text
先处理全部前驱 → 再加入当前节点
```

由此得到“前驱在前、后继在后”的合法拓扑序。若直接 `append(child)` 而不递归，只能访问一层前驱，更早的祖先会丢失。

### [Micrograd-016] 为什么反向传播使用 `reversed(topo)`

- 正向拓扑序保证前驱在后继之前
- 反向传播要求后继先把上游梯度写入当前节点，当前节点才能继续向前驱传播
- 因此必须按最终输出到叶子节点的方向执行 `_backward()`

### [Micrograd-017] `backward()` 与 `_backward()` 的区别

- `backward()`：构造拓扑序、设置最终梯度、调度整张图
- `_backward()`：单个节点的局部求导闭包
- 调度循环应调用 `node._backward()`，不能递归调用 `node.backward()`

### [Micrograd-018] 共享节点为什么要求梯度累加

对于：

```python
a = Value(3.0)
b = a * a
L = b + a
```

有：

\[
\frac{\partial L}{\partial a}=2a+1=7
\]

`a` 同时从乘法的两个输入位置和加法直连路径获得梯度贡献，因此必须使用 `+=`。

### [Micrograd-019] `tanh` 导数与局部反向规则

\[
\tanh(x)=\frac{e^{2x}-1}{e^{2x}+1}
\]

正确导数为：

\[
\tanh'(x)=\frac{4e^{2x}}{(e^{2x}+1)^2}=1-\tanh^2(x)
\]

因此局部反向规则为：

```python
self.grad += (1 - t**2) * out.grad
```

### [Micrograd-020] 闭包为什么能在外层函数结束后继续使用变量

- 内层函数会捕获它所依赖的外层变量环境
- `out._backward = _backward` 保存函数对象及其闭包
- 因此外层运算函数返回后，`_backward` 仍能访问本次运算的 `self`、`other`、`t` 和 `out`
- 保存函数时不能写 `_backward()`，否则会提前执行并把返回值 `None` 存入节点


### [Micrograd-021] `__rmul__` 的自动调用与参数绑定

- Python 对 `3 * a` 先尝试左操作数的乘法实现
- 左侧 `int` 无法处理 `Value` 时，解释器再尝试 `a.__rmul__(3)`
- 此时 `self` 指向 `a`，`other` 指向 `3`
- `return self * other` 将运算改写为 `a * 3`，进而复用 `Value.__mul__`

### [Micrograd-022] `isinstance` 与 `assert` 的职责不同

- `isinstance(obj, type)` 返回 `True/False`，负责类型判断
- `assert condition, message` 在条件为假时抛出 `AssertionError`
- 需要把普通数字转换成 `Value` 时，应使用条件判断和重新赋值，而不是 `assert`

### [Micrograd-023] 闭包捕获的是对象引用，不是属性快照

对于：

```python
y = x.exp()
```

`exp()` 内部的 `out` 与外部的 `y` 指向同一个 `Value` 对象。闭包保存的是：

```text
self → x
out  → y
```

下游节点修改 `y.grad` 后，闭包读取 `out.grad` 会看到同一个最新值。

### [Micrograd-024] 局部 `_backward` 不应声明 `self` 形参

错误：

```python
def _backward(self):
    ...
```

报错：

```text
TypeError: Value.tanh.<locals>._backward() missing 1 required positional argument: 'self'
```

局部 `_backward` 是普通闭包，应写为无参数函数；所需对象由闭包捕获。

### [Micrograd-025] 取 `.data` 会切断计算图

错误形式：

```python
t = math.exp(x.data)
```

结果是普通 `float`，不再拥有 `grad`、`label`、`_prev` 和 `_backward`。构建可微表达式时应始终使用 `Value` 运算与自定义算子。

### [Micrograd-026] 修改类后 Jupyter 仍使用旧对象

若已经定义 `exp()` 却出现：

```text
AttributeError: 'Value' object has no attribute 'exp'
```

应检查：

- `exp` 是否缩进在 `class Value` 内
- 是否误写成未被 Python 约定的 `__exp__`
- 是否重新运行完整类定义单元格
- 是否重新创建由新版 `Value` 类生成的对象

### [Micrograd-027] 常数包装判断方向写反

错误：

```python
other = other if isinstance(other, (int, float)) else Value(other)
```

这会保留普通数字，随后访问 `other.data` 时触发：

```text
AttributeError: 'int' object has no attribute 'data'
```

正确判断对象是否已经是 `Value`；不是时再包装。

### [Micrograd-028] 乘法局部导数必须使用前向值

对于：

```python
out = self * other
```

正确局部规则是：

```python
self.grad += other.data * out.grad
other.grad += self.data * out.grad
```

不能使用 `other.grad` 或刚被修改后的 `self.grad` 代替局部导数。

### [Micrograd-029] 拓扑递归必须访问 `child`

错误：

```python
for child in v._prev:
    build_topo(v)
```

`v` 已在 `visited` 中，会立即返回，导致前驱节点没有进入拓扑表。应递归处理循环得到的 `child`。

### [Micrograd-030] `__repr__` 中再次创建 `Value` 会无限递归

错误：

```python
def __repr__(self):
    return f"{Value(self.data)}"
```

格式化新对象时会再次进入同一个 `__repr__`。应直接格式化当前对象的属性。

### [Micrograd-031] 组合式 `tanh` 应复用中间节点

低效写法：

```python
((self * 2).exp() - 1) / ((self * 2).exp() + 1)
```

会创建两套重复的 \(e^{2x}\) 子图。保存一次中间结果可减少重复计算，并让同一节点的多路径梯度累积更清晰。

### [Micrograd-032] Graphviz 中出现两个数值相同的节点

在 `2 * x` 中，普通数字 `2` 会被包装成独立的常数 `Value(2)`。当 `x.data` 也恰好为 `2` 时，图中会同时出现：

- 带标签的变量节点 `x`
- 无标签的常数节点 `2`

二者数值相同，但对象身份和图中职责不同。

### [Micrograd-033] Graphviz 插件警告不影响梯度计算

```text
Could not load ... gvplugin_pango.dll
```

属于 Graphviz 字体或渲染插件警告。只要计算图能正常生成，便不影响 `Value` 引擎的前向值与反向梯度。



### [Micrograd-034] `Neuron(3)` 与 `n(x)` 为什么执行不同方法

- `Neuron(3)` 左侧是类，表示创建实例；新对象创建后执行 `__init__(self, 3)`
- `n(x)` 左侧是实例对象，Python 固定将“对象后跟括号”的语法映射到 `n.__call__(x)`
- `__init__` 通常只在创建对象时执行一次；`__call__` 可以对同一个对象执行多次

### [Micrograd-035] 类中有其他方法时，`n(x)` 是否会随机选择

不会。`n(x)` 只查找 `__call__`。例如：

```python
n.forward(x)
n.parameters()
```

必须显式调用对应名称的方法。Python 的其他语法也有固定特殊方法入口，例如 `a+b → a.__add__(b)`、`len(a) → a.__len__()`。

### [Micrograd-036] 神经元权重为什么必须是 `Value`

普通 `float` 没有：

```text
.grad
._prev
._backward
```

若权重和偏置保存为 `float`，它们无法作为 Micrograd 参数参与完整计算图和反向传播。当前项目使用的是自定义 `Value`，不是 PyTorch `Tensor`。

### [Micrograd-037] 随机初始化不是为了直接防止过拟合

随机初始化的直接作用是打破对称性。若多个神经元参数完全相同，它们会产生相同输出、得到相同梯度并保持相同更新轨迹，相当于重复同一个神经元。

### [Micrograd-038] 偏置只能在点积后加入一次

正确形式：

\[
act=\sum_i w_i x_i+b
\]

错误形式：

\[
\sum_i(w_i x_i+b)
\]

后者会把偏置重复加入 `nin` 次。使用 `sum(generator, self.b)` 时，`self.b` 已是累计初值，无需在结果后再次加偏置。

### [Micrograd-039] `sum(generator, start)` 的括号要求

当生成器表达式与 `sum` 的第二个参数同时出现时，应写：

```python
sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
```

生成器需要单独括起来，否则逗号可能被解析到生成器表达式内部并触发语法错误。

### [Micrograd-040] `zip` 的静默截断风险

`zip(self.w, x)` 会在较短一侧结束，不一定报错。若神经元有 3 个权重而输入只有 2 个，第三个权重不会参与当前前向图，其梯度也会保持为 `0.0`。这种静默漏算通常比直接报错更危险。

### [Micrograd-041] `Layer` 中不能用列表乘法复制神经元

```python
[Neuron(nin)] * nout
```

只创建一个神经元，再复制其引用。正确方式是列表推导式，每轮重新执行 `Neuron(nin)`，得到独立对象。

### [Micrograd-042] `Layer` 参数数量必须包含偏置

对于 `Layer(nin, nout)`：

\[
\text{参数量}=nout\times(nin+1)
\]

`Layer(5, 3)` 不是 15 个参数，而是 15 个权重加 3 个偏置，共 18 个 `Value` 参数。

### [Micrograd-043] 单神经元层当前仍返回列表

当前 `Layer.__call__` 始终 `return outs`，因此：

```python
Layer(3, 1)(x)
```

返回 `[Value(...)]`。列表没有 `backward()`；真正的输出节点是 `outs[0]`。保持始终返回列表可确保接口类型稳定。

### [Micrograd-044] 参数是否获得梯度取决于损失图可达性

同属一个 `Layer` 不代表全部参数都一定获得梯度。只有从当前 `loss` 反向遍历能够到达的节点才会执行局部 `_backward()`。未参与当前损失路径的参数梯度通常保持 `0.0`，而不是 `1`。


### [Micrograd-045] `sizes` 为什么是 `[nin] + nouts`

`[nin]` 将单个输入维度包装成单元素列表，使其能与 `nouts` 拼接。顺序必须从网络输入开始，例如 `[3] + [4,4,1] = [3,4,4,1]`，才能依次构造 `3→4`、`4→4`、`4→1`。

### [Micrograd-046] 构造 `self.layers` 时圆括号、方括号与循环边界

创建层对象应写：

```python
Layer(sizes[i], sizes[i + 1])
```

`Layer[...]` 是下标/泛型语法，不是调用构造函数。循环应使用：

```python
range(len(sizes) - 1)
```

否则最后一次会访问不存在的 `sizes[len(sizes)]`。

### [Micrograd-047] 为什么只创建 `MLP` 没有输出

```python
model = MLP(...)
```

只执行模型初始化并把对象绑定到变量，不会自动前向传播，也不会打印结果。必须显式调用：

```python
out = model(x)
```

Jupyter 中将 `out` 放在单元格最后一行可自动显示；普通脚本中需 `print(out)`。

### [Micrograd-048] `MLP.__call__` 中局部 `x` 与外部 `x`

二者是不同作用域中的变量，函数刚进入时暂时引用同一个输入对象。每次 `x = layer(x)` 都会让局部变量重新绑定到新输出列表，外部输入变量不会被改写。

### [Micrograd-049] 全连接为什么让每个隐藏输出连接所有下一层神经元

`Layer(nin, nout)` 创建 `nout` 个 `Neuron(nin)`。每个神经元都有 `nin` 个独立权重，并接收完整输入向量，因此上一层每个输出都会通过不同权重连接到每个下一层神经元。

### [Micrograd-050] 只选择一个最终输出时，隐藏层为何仍可能全部获得梯度

即使 `loss = out[0]` 裁掉了其他最终输出分支，`out[0]` 仍由上一层的全部隐藏输出共同计算。因此这些隐藏节点及其参数仍属于当前损失的祖先图，都会参与反向传播；某个数值梯度仍可能在特殊情况下恰好为 0。

### [Micrograd-051] `tanh` 是否属于激活函数

是。当前神经元执行 `out = tanh(w·x+b)`。激活函数的本质作用是提供非线性，不是只负责“激活隐藏层”。隐藏层通常需要非线性，输出层激活则取决于任务。

### [Micrograd-052] 如何绘制所有并列输出的完整计算图

`draw_dot` 一次只接受一个根节点。将所有输出通过加法合成一个标量根节点，再对该根执行一次 `backward()` 与 `draw_dot()`，即可覆盖全部输出分支。不要为了可视化而依次对每个输出重复调用 `backward()`，否则参数梯度会跨多次反传累积。


## 🔁 当前学习方法

采用以下循环，而不是看完整个视频后再集中照写：

```text
观看 10～20 分钟或一个完整知识点
→ 遇到不理解的设计立即提问
→ 关闭参考代码并独立复现
→ 先预测结果，再运行最小实验
→ 对照视频检查和修正
→ 完成一个稳定小里程碑后提交
```

原则：

- 自己手写，不直接复制完整源码
- 一次只实现一个小功能
- 每个算子先理解前向语义，再学习局部梯度
- 报错时优先定位原因，不直接替换成完整答案
- 每个里程碑都用最小实验验证

## 🧭 后续里程碑

- [x] M1：`Value` 基础结构
- [x] M1：实现 `__repr__`
- [x] M1：实现加法和乘法前向计算
- [x] M1：保存 `_prev` 和 `_op`
- [x] M1：验证基础前向计算图
- [x] M2：理解并加入 `label`、`grad` 的设计
- [x] M2：配置 Graphviz，并理解 `trace` / `draw_dot`
- [x] M3：手工推导加法和乘法的局部梯度
- [x] M3：理解闭包并为算子保存 `_backward`
- [x] M3：理解默认空函数 `lambda: None`
- [x] M3：理解上游梯度 `out.grad` 的逐层写入机制
- [x] M3：理解共享节点与多路径梯度必须使用 `+=`
- [x] M4：理解基于 `_prev` 与 `visited` 的 DFS 拓扑排序
- [x] M4：理解统一 `backward()` 的完整调度结构
- [x] M4：完成拓扑顺序、反向顺序和分支梯度的手算练习
- [x] M4：在 Notebook 中独立复现并运行统一 `backward()`
- [ ] M4：将最新代码同步到 Google Drive Notebook
- [x] M5：理解 `tanh` 的前向公式与局部反向规则
- [x] M5：在 Notebook 中独立实现并验证组合式 `tanh`
- [x] M5：支持普通常数参与加法、乘法等运算
- [x] M5：实现 `exp`、幂运算、减法和除法等算子
- [x] M6：实现 `Neuron`
- [x] M6：实现 `Layer`
- [x] M6：实现 `MLP`
- [x] M6：理解全连接层的完整连接关系与层间维度传递
- [x] M6：使用汇总根节点绘制多输出 MLP 的完整计算图
- [x] M6：理解隐藏层激活与输出层激活的设计边界
- [ ] M7：实现损失函数和训练循环
- [ ] M5：使用 PyTorch API 构造同一表达式并对照梯度
- [ ] M7：与 PyTorch 梯度结果进行完整对照测试

## 📝 日志维护约定

本项目日志统一使用：

```text
micrograd_learning_log.md
```

与 D2L 的主学习日志：

```text
study_log.md
```

保持分离。

职责划分：

- `study_log.md`：记录 D2L 章节进度、PyTorch API、理论问题和章节 Bug
- `micrograd_learning_log.md`：记录 Micrograd 实现里程碑、自动微分设计、代码问题和测试结果
- `projects/micrograd/README.md`：对外说明项目目标、运行方式、目录结构和最终功能
