# math.c 程序 README

## 程序简介
`math.c` 是一个功能丰富的数学函数库，提供了多种数学运算和算法的实现。它涵盖了基本数学运算（如加减乘除、阶乘、幂运算等）、数组操作（如求和、求积、排序等）、矩阵计算（如行列式、逆矩阵、特征值等）、数值积分、随机数生成、复数运算等。此外，还实现了多种高级数学算法，如 QR 分解、奇异值分解（SVD）、幂迭代法、Arnoldi 迭代等。

## 文件结构
- **math.c**：包含所有数学函数的实现。
- **math.h**：头文件，声明了 `math.c` 中定义的函数和数据结构。

## 编译与运行
### 编译
确保你已经安装了 `clang` 或其他支持 C 语言的编译器。在终端中运行以下命令来编译程序：
```bash
clang -fprofile-instr-generate -fcoverage-mapping -fcoverage-mcdc -fno-inline -O0 -c math.c -o math.o
```
这将生成目标文件 `math.o`。

### 链接与运行
将 `math.o` 链接到你的主程序中。例如，如果你有一个测试程序 `test.c`，可以使用以下命令进行链接和运行：
```bash
clang++ -fprofile-instr-generate -fcoverage-mapping -fcoverage-mcdc -o test test.o math.o
./test
```
这将运行测试程序并输出结果。

## 功能模块
### 基本数学运算
- 加法、减法、乘法、除法
- 阶乘、幂运算、模运算
- 绝对值、平方根、立方根等

### 数组操作
- 求和、求积、求最大值、求最小值
- 排序、二分查找、线性查找
- 数组的均值、方差、标准差等统计量计算

### 矩阵计算
- 行列式、逆矩阵、转置矩阵
- 特征值、特征向量、特征多项式
- 矩阵分解（QR 分解、LU 分解、Cholesky 分解等）
- 矩阵的幂、指数函数、对数函数
- 矩阵的范数、条件数、谱半径
- 矩阵的乘积、Kronecker 积、Hadamard 积等

### 数值积分
- 梯形法则、辛普森法则

### 随机数生成
- 均匀分布随机数、正态分布随机数

### 复数运算
- 复数加法、减法、乘法、除法
- 复数的幂运算、开方

### 向量运算
- 点积、叉积、向量归一化

### 线性方程组求解
- 高斯消元法

### 特征值与特征向量
- 幂迭代法、反幂迭代法、QR 算法

### 矩阵的 Hessenberg 形式
- Householder 变换、Arnoldi 迭代

### Givens 旋转与 Householder 变换
- 矩阵的 Givens 旋转、Householder 变换

## 使用示例
以下是一些函数的使用示例：
```c
#include "math.h"

int main() {
    // 基本数学运算
    int sum = add(3, 5);
    printf("3 + 5 = %d\n", sum);

    // 数组操作
    int array[] = {1, 2, 3, 4, 5};
    int size = sizeof(array) / sizeof(array[0]);
    int max_value = find_max(array, size);
    printf("Max value in array: %d\n", max_value);

    // 矩阵计算
    double matrix[2][2] = {{1, 2}, {3, 4}};
    double determinant = matrix_determinant(2, matrix);
    printf("Determinant of matrix: %f\n", determinant);

    // 数值积分
    double integral = trapezoidal_rule(sin, 0, M_PI, 100);
    printf("Integral of sin(x) from 0 to pi: %f\n", integral);

    return 0;
}
```

## 注意事项
- 程序中部分函数（如矩阵的逆矩阵、特征值等）的实现假设输入矩阵是方阵。
- 在使用数值算法时，需要注意算法的收敛性和数值稳定性。
- 部分函数（如复数运算、矩阵的 Hessenberg 形式等）需要额外的数学背景知识来正确理解和使用。

## 贡献与反馈
欢迎对程序进行改进和扩展。如果你发现任何问题或有改进建议，请随时提出。
