# Numpy

## 安装

```shell
pip3 install --user numpy scipy matplotlib
```

## 基础知识

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

