# Задача 1
Дано: три последовательных умножения матриц на вектор (на примере одного обучающего вектора $x$):

$z^1 = w^1 a^0, \quad z^2 = w^2 z^1, \quad z^3 = w^3 z^2$

Здесь $w^1, w^2, w^3$ — двумерные массивы весов, а $a^0$ — одномерный массив входных данных.

Предположим, что $n^0$, $n^1$ и $n^2$ - количество нейронов в слоях 0, 1 и 2. Написать формулу со знаками $\sum$ (аналог вложенных циклов $for$), которая вычисляет один конкретный элемент $z_{i}^3$ напрямую через элементы массива $a_{l}^0$ и элементы матриц $w^1, w^2, w^3$.

<i> <b>Правильный ответ для покомпонентной записи: </b>
<div align="center">
  
  $z_i^3 = \sum_{j=1}^{n^2} \left[ w_{ij}^3 \cdot \sum_{k=1}^{n^1} \left[ w_{jk}^2 \cdot \sum_{l=1}^{n^0} w_{kl}^1 a_l^0 \right] \right]$

</div>

<b> Правильный ответ для матричной записи: </b>

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

Правильный ответ для расчёта данного выражения в Python:
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

Здесь $f^1$ и $g^2$ — функции над массивами, в которых $каждый$ выходной элемент нейрона зависит от $всех$ его входных элементов (как, например, в SoftMax).

Требуется сначала написать формулу со вложенными суммами $\sum$ для производной $\frac{\partial C_x}{\partial z_{xp}^1}$, используя частные производные на каждом шаге: $\frac{\partial C_x}{\partial z_{xk}^2}$, $\frac{\partial z_{xk}^2}{\partial a_{xr}^1}$ и $\frac{\partial a_{xr}^1}{\partial z_{xp}^1}$.

Далее нужно собрать все получившиеся таблицы частных производных в двумерные матрицы (а где-то, возможно, и в столбцы) и перевести их в векторный вид для будущих циклов. Будьте внимательны с транспонированием, если оно потребуется.

<i> <b>Правильный ответ для покомпонентной записи: </b>
<div align="center">
  
  $\frac{\partial C}{\partial z_p^1} = \sum_{j=1}^{n^2} \left[ \frac{\partial C}{\partial z_j^2} \cdot \left[ \sum_{k=1}^{n^1} \left[ \frac{\partial z_j^2}{\partial a_k^1} \cdot \frac{\partial a_k^1}{\partial z_p^1} \right] \right] \right]$

</div>

<b> Правильный ответ для матричной записи: </b>

```math
\(\frac{\partial C}{\partial z_p^1} = \begin{pmatrix} 1_1 & 1_2 & \dots & 1_{n^2} \end{pmatrix} \cdot \left[ \begin{pmatrix} \frac{\partial C}{\partial z_1^2} \\  \frac{\partial C}{\partial z_2^2} \\  \dots \\  \frac{\partial C}{\partial z_{n^2}^2} \end{pmatrix} \odot \left[ \begin{pmatrix} \frac{\partial z_1^2}{\partial a_1^1} & \frac{\partial z_2^2}{\partial a_1^1} & \dots & \frac{\partial z_{n^2}^2}{\partial a_1^1} \\  \frac{\partial z_1^2}{\partial a_2^1} & \frac{\partial z_2^2}{\partial a_2^1} & \dots & \frac{\partial z_{n^2}^2}{\partial a_2^1} \\  \dots & \dots & \ddots & \dots \\  \frac{\partial z_1^2}{\partial a_{n^1}^1} & \frac{\partial z_2^2}{\partial a_{n^1}^1} & \dots & \frac{\partial z_{n^2}^2}{\partial a_{n^1}^1} \end{pmatrix}^T \cdot \begin{pmatrix} \frac{\partial a_1^1}{\partial z_p^1} & \frac{\partial a_2^1}{\partial z_p^1} & \dots & \frac{\partial a_{n^1}^1}{\partial z_p^1} \end{pmatrix}^T \right] \right]\)
```


Правильный ответ для расчёта данного выражения в Python:
</i>

```python
import numpy as np

def feedforward_linear(a, weights):
    for w in weights:
        a = np.dot(w, a)
    return a
```

