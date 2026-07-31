# Micrograd 学习日志

## 📌 当前进度

- **最近更新时间**：2026-08-01
- **项目名称**：Micrograd From Scratch
- **当前阶段**：基础标量自动微分引擎已独立复现；已用基础算子组合实现 `tanh`，下一步进入 PyTorch API 对照学习
- **当前里程碑**：M5（已完成）— 已实现常数自动包装、反向运算符、减法、幂、除法、`exp`、统一 `backward()` 与组合式 `tanh`，并通过计算图和数值结果验证
- **代码状态**：当前本地 Notebook 中的最小 `Value` 引擎已可运行；本次仅生成更新后的日志文件，未上传或覆盖 Google Drive 文件

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
- [ ] M6：实现 `Neuron`
- [ ] M6：实现 `Layer`
- [ ] M6：实现 `MLP`
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
