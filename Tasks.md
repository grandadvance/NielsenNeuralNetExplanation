# Problem 1
Given: three consecutive matrix-vector multiplications (using a single training vector $x$ as an example):

$z^1 = w^1 a^0, \quad z^2 = w^2 z^1, \quad z^3 = w^3 z^2$

Here $w^1, w^2, w^3$ - are 2D weight arrays, and $a^0$ - is a 1D input data array.

Suppose that $n^0$, $n^1$ and $n^2$ - are the number of neurons in layers 0, 1 and 2. Write a formula using $\sum$ signs (similar to nested $for$ loops), that calculates one specific element $z_{i}^3$ directly using the elements of the array $a_{l}^0$ and the elements of the matrices $w^1, w^2, w^3$.

<i> <b> Correct answer for element-wise notation: </b>
<div align="center">
  
  $z_i^3 = \sum_{j=1}^{n^2} \left[ w_{ij}^3 \cdot \sum_{k=1}^{n^1} \left[ w_{jk}^2 \cdot \sum_{l=1}^{n^0} w_{kl}^1 a_l^0 \right] \right]$

</div>

<b> Correct answer for matrix notation: </b>

$$
z_i^3 = 
\begin{pmatrix} 
w_{i\,1}^3 & w_{i\,2}^3 & \dots & w_{i\,n^2}^3 
\end{pmatrix} 
\cdot 
\begin{pmatrix} 
w_{1\,1}^2 & w_{1\,2}^2 & \dots & w_{1\,n^1}^2 \\\\ 
w_{2\,1}^2 & w_{2\,2}^2 & \dots & w_{2\,n^1}^2 \\\\ 
\dots & \dots & \ddots & \dots \\\\ 
w_{n^2\,1}^2 & w_{n^2\,2}^2 & \dots & w_{n^2\,n^1}^2 
\end{pmatrix} 
\cdot 
\begin{pmatrix} 
w_{1\,1}^1 & w_{1\,2}^1 & \dots & w_{1\,n^0}^1 \\\\ 
w_{2\,1}^1 & w_{2\,2}^1 & \dots & w_{2\,n^0}^1 \\\\ 
\dots & \dots & \ddots & \dots \\\\ 
w_{n^1\,1}^1 & w_{n^1\,2}^1 & \dots & w_{n^1\,n^0}^1 
\end{pmatrix} 
\cdot 
\begin{pmatrix} 
a_1^0 \\\\ 
a_2^0 \\\\ 
\dots \\\\ 
a_{n^0}^0 
\end{pmatrix}
$$

Correct answer for calculating this expression in Python:
</i>

```python
import numpy as np

def feedforward_linear(a, weights):
    for w in weights:
        a = np.dot(w, a)
    return a
```

# Задача 2
Дано: цепочка вычислений на двух слоях для одного обучающего примера. Одна функция от другой зависит так:

$a^1 = f^1(z^1), \quad z^2 = g^2(a^1), \quad C = h(z^2)$

Здесь $f^1$ и $g^2$ — функции над массивами, в которых <b>каждый</b> выходной элемент нейрона зависит от <b>всех</b> его входных элементов (как, например, в <b>SoftMax</b>).

Требуется сначала написать формулу со вложенными суммами $\sum$ для производной $\frac{\partial C_x}{\partial z_{xp}^1}$, используя частные производные на каждом шаге: $\frac{\partial C_x}{\partial z_{xk}^2}$, $\frac{\partial z_{xk}^2}{\partial a_{xr}^1}$ и $\frac{\partial a_{xr}^1}{\partial z_{xp}^1}$.

Далее нужно собрать все получившиеся таблицы частных производных в двумерные матрицы (а где-то, возможно, и в столбцы) и перевести их в векторный вид для будущих циклов. Будьте внимательны с транспонированием, если оно потребуется.

<i> <b>Правильный ответ для покомпонентной записи: </b>
<div align="center">
  
  $\frac{\partial C}{\partial z_p^1} = \sum_{j=1}^{n^2} \left[ \frac{\partial C}{\partial z_j^2} \cdot \left[ \sum_{k=1}^{n^1} \left[ \frac{\partial z_j^2}{\partial a_k^1} \cdot \frac{\partial a_k^1}{\partial z_p^1} \right] \right] \right]$

</div>

<b> Правильный ответ для матричной записи: </b>

```math
\frac{\partial C}{\partial z_p^1} = \begin{pmatrix} 1_1 & 1_2 & \dots & 1_{n^2} \end{pmatrix} \cdot \left[ \begin{pmatrix} \frac{\partial C}{\partial z_1^2} \\  \frac{\partial C}{\partial z_2^2} \\  \dots \\  \frac{\partial C}{\partial z_{n^2}^2} \end{pmatrix} \odot \left[ \begin{pmatrix} \frac{\partial z_1^2}{\partial a_1^1} & \frac{\partial z_2^2}{\partial a_1^1} & \dots & \frac{\partial z_{n^2}^2}{\partial a_1^1} \\  \frac{\partial z_1^2}{\partial a_2^1} & \frac{\partial z_2^2}{\partial a_2^1} & \dots & \frac{\partial z_{n^2}^2}{\partial a_2^1} \\  \dots & \dots & \ddots & \dots \\  \frac{\partial z_1^2}{\partial a_{n^1}^1} & \frac{\partial z_2^2}{\partial a_{n^1}^1} & \dots & \frac{\partial z_{n^2}^2}{\partial a_{n^1}^1} \end{pmatrix}^T \cdot \begin{pmatrix} \frac{\partial a_1^1}{\partial z_p^1} & \frac{\partial a_2^1}{\partial z_p^1} & \dots & \frac{\partial a_{n^1}^1}{\partial z_p^1} \end{pmatrix}^T \right] \right]
```


Правильный ответ для расчёта данного выражения в Python:
</i>

```python
import numpy as np

def compute_dC_dz1_p(dC_dz2, dz2_da1, da1_dz1_p):

    # Аргументы:
    # dC_dz2    - вектор-столбец размерности (n^2, 1)
    # dz2_da1   - матрица Якоби размерности (n^1, n^2)
    # da1_dz1_p - вектор-строка размерности (1, n^1) 

    n_2 = dC_dz2.shape[0]
    ones_row = np.ones((1, n_2))  # Строка (1_1, 1_2, ..., 1_{n^2}) размерности (1, n^2)

    # Внутренние скобки: (dz^2 / da^1)^T @ (da^1 / dz^1_p)^T
    inner = np.dot(dz2_da1.transpose(), da1_dz1_p.transpose())

    # Поэлементное умножение (*) двух столбцов с размерностями (n^2, 1)
    hadamard = dC_dz2 * inner

    # Умножение строки единиц (1, n^2) на столбец (n^2, 1)
    result = np.dot(ones_row, hadamard)

    # Извлекаем конкретное число из массива формы (1, 1)
    return float(result[0, 0])

```

