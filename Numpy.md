# Numpy

## 安装

```shell
pip3 install --user numpy scipy matplotlib
```



## 基础知识

### 功能方法概述

以下是按类别排序的一些有用的NumPy函数和方法名称的列表。有关完整列表，请参阅[参考手册](https://www.numpy.org.cn/reference/)里的[常用API](https://www.numpy.org.cn/reference/routines/)。

- **数组的创建（Array Creation）** - [arange](https://numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange), [array](https://numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array), [copy](https://numpy.org/devdocs/reference/generated/numpy.copy.html#numpy.copy), [empty](https://numpy.org/devdocs/reference/generated/numpy.empty.html#numpy.empty), [empty_like](https://numpy.org/devdocs/reference/generated/numpy.empty_like.html#numpy.empty_like), [eye](https://numpy.org/devdocs/reference/generated/numpy.eye.html#numpy.eye), [fromfile](https://numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile), [fromfunction](https://numpy.org/devdocs/reference/generated/numpy.fromfunction.html#numpy.fromfunction), [identity](https://numpy.org/devdocs/reference/generated/numpy.identity.html#numpy.identity), [linspace](https://numpy.org/devdocs/reference/generated/numpy.linspace.html#numpy.linspace), [logspace](https://numpy.org/devdocs/reference/generated/numpy.logspace.html#numpy.logspace), [mgrid](https://numpy.org/devdocs/reference/generated/numpy.mgrid.html#numpy.mgrid), [ogrid](https://numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid), [ones](https://numpy.org/devdocs/reference/generated/numpy.ones.html#numpy.ones), [ones_like](https://numpy.org/devdocs/reference/generated/numpy.ones_like.html#numpy.ones_like), [zeros](https://numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros), [zeros_like](https://numpy.org/devdocs/reference/generated/numpy.zeros_like.html#numpy.zeros_like)
- **转换和变换（Conversions）** - [ndarray.astype](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype), [atleast_1d](https://numpy.org/devdocs/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d), [atleast_2d](https://numpy.org/devdocs/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d), [atleast_3d](https://numpy.org/devdocs/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d), [mat](https://numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat)
- **操纵术（Manipulations）** - [array_split](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split), [column_stack](https://numpy.org/devdocs/reference/generated/numpy.column_stack.html#numpy.column_stack), [concatenate](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate), [diagonal](https://numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal), [dsplit](https://numpy.org/devdocs/reference/generated/numpy.dsplit.html#numpy.dsplit), [dstack](https://numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack), [hsplit](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit), [hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack), [ndarray.item](https://numpy.org/devdocs/reference/generated/numpy.ndarray.item.html#numpy.ndarray.item), [newaxis](https://www.numpy.org.cn/reference/constants.html#numpy.newaxis), [ravel](https://numpy.org/devdocs/reference/generated/numpy.ravel.html#numpy.ravel), [repeat](https://numpy.org/devdocs/reference/generated/numpy.repeat.html#numpy.repeat), [reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape), [resize](https://numpy.org/devdocs/reference/generated/numpy.resize.html#numpy.resize), [squeeze](https://numpy.org/devdocs/reference/generated/numpy.squeeze.html#numpy.squeeze), [swapaxes](https://numpy.org/devdocs/reference/generated/numpy.swapaxes.html#numpy.swapaxes), [take](https://numpy.org/devdocs/reference/generated/numpy.take.html#numpy.take), [transpose](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose), [vsplit](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit), [vstack](https://numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack)
- **询问（Questions）** - [all](https://numpy.org/devdocs/reference/generated/numpy.all.html#numpy.all), [any](https://numpy.org/devdocs/reference/generated/numpy.any.html#numpy.any), [nonzero](https://numpy.org/devdocs/reference/generated/numpy.nonzero.html#numpy.nonzero), [where](https://numpy.org/devdocs/reference/generated/numpy.where.html#numpy.where),
- **顺序（Ordering）** - [argmax](https://numpy.org/devdocs/reference/generated/numpy.argmax.html#numpy.argmax), [argmin](https://numpy.org/devdocs/reference/generated/numpy.argmin.html#numpy.argmin), [argsort](https://numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort), [max](https://docs.python.org/dev/library/functions.html#max), [min](https://docs.python.org/dev/library/functions.html#min), [ptp](https://numpy.org/devdocs/reference/generated/numpy.ptp.html#numpy.ptp), [searchsorted](https://numpy.org/devdocs/reference/generated/numpy.searchsorted.html#numpy.searchsorted), [sort](https://numpy.org/devdocs/reference/generated/numpy.sort.html#numpy.sort)
- **操作（Operations）** - [choose](https://numpy.org/devdocs/reference/generated/numpy.choose.html#numpy.choose), [compress](https://numpy.org/devdocs/reference/generated/numpy.compress.html#numpy.compress), [cumprod](https://numpy.org/devdocs/reference/generated/numpy.cumprod.html#numpy.cumprod), [cumsum](https://numpy.org/devdocs/reference/generated/numpy.cumsum.html#numpy.cumsum), [inner](https://numpy.org/devdocs/reference/generated/numpy.inner.html#numpy.inner), [ndarray.fill](https://numpy.org/devdocs/reference/generated/numpy.ndarray.fill.html#numpy.ndarray.fill), [imag](https://numpy.org/devdocs/reference/generated/numpy.imag.html#numpy.imag), [prod](https://numpy.org/devdocs/reference/generated/numpy.prod.html#numpy.prod), [put](https://numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put), [putmask](https://numpy.org/devdocs/reference/generated/numpy.putmask.html#numpy.putmask), [real](https://numpy.org/devdocs/reference/generated/numpy.real.html#numpy.real), [sum](https://numpy.org/devdocs/reference/generated/numpy.sum.html#numpy.sum)
- **基本统计（Basic Statistics）** - [cov](https://numpy.org/devdocs/reference/generated/numpy.cov.html#numpy.cov), [mean](https://numpy.org/devdocs/reference/generated/numpy.mean.html#numpy.mean), [std](https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std), [var](https://numpy.org/devdocs/reference/generated/numpy.var.html#numpy.var)
- **基本线性代数（Basic Linear Algebra）** - [cross](https://numpy.org/devdocs/reference/generated/numpy.cross.html#numpy.cross), [dot](https://numpy.org/devdocs/reference/generated/numpy.dot.html#numpy.dot), [outer](https://numpy.org/devdocs/reference/generated/numpy.outer.html#numpy.outer), [linalg.svd](https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd), [vdot](https://numpy.org/devdocs/reference/generated/numpy.vdot.html#numpy.vdot)





### 属性

Numpy的数据类被调用 `ndarray` 

* **ndarray.ndim** - 维度
* **ndarray.shape** - 数组的维度。例如对于n行m列的矩阵，那么`shape` = `(n,m)`，因此shape数据的长度就是`ndim`。
* **ndarray.size** - 数组元素的总数。等于shape的乘积。
* **ndarray.dtype** - 数组中元素的类型。
* **ndarray.itemsize** - 每个元素的字节大小。
* **ndarray.data** - 一般不用，就是实际元素

### 创建数组

```python
import numpy as np

# type 1: 元素已知
a = np.array([2, 3, 4])
b = np.array([(1, 2, 3), (4, 5, 6)], dtype=complex) # 显示指定类型

# type 2: 元素位置 shape已知
np.zeros((3, 4))
np.ones((3, 4), dtype=complex)
np.empty((2, 3)) # 未初始化

# type 3: 使用range 创建数字矩阵 可以设置步长；使用linspace，在var1和var2之间取多少个值
np.arange(10, 40, 5, dtype=complex).reshape(2, 3)
np.linspace(0, 2 , 9)
```



### 基本操作

* 所有的基本运算 + - * / **等

* 矩阵乘法使用`@`,例如a@b；或者a.dot(b)

* 使用`+= *=`会直接更改矩阵，而不会产生新的矩阵

* ```python
  # 常见的方法
  a = np.random.random((2, 3))
  a.sum()
  a.min()
  a.max()
  ```

* `exp sqrt add cos sin`一类通函数

![image-20211126224847427](C:\Users\86150\AppData\Roaming\Typora\typora-user-images\image-20211126224847427.png)



### 索引、切片、迭代

对于**一维数组**的索引就和python普通的一样一样的

```python
a = np.arange(12)
# 从0索引到6，步长为2
a[0:6:2] = -1000
```

**多维数组**，每一维都和一维是一样的，每一个维度用`,`分割开

```python
def f(x, y):
    return 10*x + y

b np.fromfunction(f, (5, 4),dtype)

# 0-4行的第一列
b[0:5, 1]
# ... 代替剩余的所有维度使用:
b[0:5, ...]
```

迭代使用`flat`属性

```python
# 这样是相对于第一个轴完成的
for row in b:
    print(row)
    
# 使用flat
for element in b.flat:
    print(element)
```



### 形状操纵

#### 改变形状

```python
a.ravel() # 摊平
a.reshape(6, 2) # 改变shape,不改变数组本身
a.T # 转置
a.resize((6, 2)) # 改变shape，改变数组本身
```

#### 将数组叠加

```python
# 沿着第一轴堆叠
np.vstack((a, b))

# 沿着第二轴堆叠
np.hstack((a, b))
```

#### 拆分数组

`hsplit`用于拆分数组，沿着水平轴拆分

`vsplit`用于拆分数组，沿着垂直轴分割

`array_split`允许指定要分割的轴

```python
a = np.floor(10*np.random.random((2,12)))
# 将a拆分成三个
np.hsplit(a,3)
```



### 拷贝和视图

#### 完全不复制

```python
a = np.arange(12)
b = a 
# b is a, b 就是a的一个别名罢了
```

#### 视图或浅拷贝

```python
c = a.view()
# c is a ? false 

c.base = a
# true

# 但是只是浅拷贝，数据还是共享的，也就是如果改变c的数据，那么a的数据也会改变，但是改变c的形状是可以的。a不会跟着改变

# 切片就是返回一个视图
```

#### 深拷贝

```python
d = a.copy()
# 此时就完全不一样了，数据也是单独的。

# 因此如果切片后如果不再需要改变原始数组，应该切片后调用copy
a = np.arange(100)
b = a[:20].copy()
del a 
```



### 花式索引

使用索引数组进行索引



## Numpy遇到的神奇东西

### 如何交换两行

```python
# 以交换第二行和第三行为例子
# 1.最直接的方法,使用深拷贝；如果不使用深拷贝，那么这个tmp只是a[1]的别名而已。
a = np.zeros(9,9)
tmp = np.copy(a[1])
a[1] = a[2]
a[2] = a[1]

# 2.
a[[1,2],:] = a[[2, 1],:]
```

