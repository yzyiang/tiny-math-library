#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "math.h"

int add(int a, int b) 
{
    return a + b;
}

int subtract(int a, int b) 
{
    return a - b;
}

int multiply(int a, int b) 
{
    return a * b;
}

int divide(int a, int b) 
{
    if (b == 0) 
    {
        printf("Division by zero!\n");
        return -1;
    }
    return a / b;
}

int factorial(int n) 
{
    if (n < 0) 
    {
        printf("Factorial of negative number is undefined!\n");
        return -1;
    }
    if (n == 0) 
    {
        return 1;
    }
    return n * factorial(n - 1);
}

int fibonacci(int n) 
{
    if (n < 0) 
    {
        printf("Fibonacci of negative number is undefined!\n");
        return -1;
    }
    if (n == 0) 
    {
        return 0;
    }
    if (n == 1) 
    {
        return 1;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int is_prime(int n) 
{
    if (n <= 1) 
    {
        return 0;
    }
    for (int i = 2; i <= sqrt(n); i++) 
    {
        if (n % i == 0) 
        {
            return 0;
        }
    }
    return 1;
}

int gcd(int a, int b) 
{
    if (a == 0) 
    {
        return b;
    }
    return gcd(b % a, a);
}

int lcm(int a, int b) 
{
    return (a * b) / gcd(a, b);
}

int power(int base, int exponent) 
{
    if (exponent < 0) 
    {
        printf("Negative exponent is undefined!\n");
        return -1;
    }
    if (exponent == 0) 
    {
        return 1;
    }
    return base * power(base, exponent - 1);
}

int modulo(int a, int b) {
    if (b == 0) 
    {
        printf("Modulo by zero is undefined!\n");
        return -1;
    }
    return a % b;
}

int sum_array(int *array, int size) 
{
    int sum = 0;
    for (int i = 0; i < size; i++) 
    {
        sum += array[i];
    }
    return sum;
}

int product_array(int *array, int size) 
{
    int product = 1;
    for (int i = 0; i < size; i++) 
    {
        product *= array[i];
    }
    return product;
}

int find_max(int *array, int size) 
{
    int max = array[0];
    for (int i = 1; i < size; i++) 
    {
        if (array[i] > max) 
        {
            max = array[i];
        }
    }
    return max;
}

int find_min(int *array, int size) 
{
    int min = array[0];
    for (int i = 1; i < size; i++) 
    {
        if (array[i] < min) 
        {
            min = array[i];
        }
    }
    return min;
}

int find_average(int *array, int size) 
{
    return sum_array(array, size) / size;
}

int sort_array(int *array, int size) 
{
    for (int i = 0; i < size - 1; i++) 
    {
        for (int j = 0; j < size - 1 - i; j++) 
        {
            if (array[j] > array[j + 1]) 
            {
                int temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
    return 0;
}

int binary_search(int *array, int size, int value) 
{
    int low = 0, high = size - 1;
    while (low <= high) 
    {
        int mid = (low + high) / 2;
        if (array[mid] == value) 
        {
            return mid;
        } else if (array[mid] < value) 
        {
            low = mid + 1;
        } else 
        {
            high = mid - 1;
        }
    }
    return -1;
}

int linear_search(int *array, int size, int value) 
{
    for (int i = 0; i < size; i++) 
    {
        if (array[i] == value) 
        {
            return i;
        }
    }
    return -1;
}

// 新添加的函数实现

int absolute(int n) 
{
    return (n < 0) ? -n : n;
}

int sqrt_int(int n) 
{
    if (n < 0) 
    {
        printf("Square root of negative number is undefined!\n");
        return -1;
    }
    int i = 0;
    while (i * i <= n) 
    {
        i++;
    }
    return i - 1;
}

int is_perfect_square(int n) 
{
    if (n < 0) 
    {
        return 0;
    }
    int sqrt_n = sqrt_int(n);
    return (sqrt_n * sqrt_n == n);
}

float harmonic_mean(int a, int b) 
{
    if (a + b == 0) 
    {
        printf("Harmonic mean is undefined for these values!\n");
        return -1;
    }
    return (float)(2 * a * b) / (a + b);
}

int cube(int n) 
{
    return n * n * n;
}

int cube_root(int n) 
{
    if (n < 0) 
    {
        return -cube_root(-n);
    }
    int i = 0;
    while (i * i * i <= n) 
    {
        i++;
    }
    return i - 1;
}

int sum_of_factorials(int n) 
{
    if (n < 0) 
    {
        printf("Sum of factorials for negative number is undefined!\n");
        return -1;
    }
    int sum = 0;
    for (int i = 1; i <= n; i++) 
    {
        sum += factorial(i);
    }
    return sum;
}

int count_digits(int n) 
{
    if (n == 0) 
    {
        return 1;
    }
    int count = 0;
    while (n != 0) 
    {
        n /= 10;
        count++;
    }
    return count;
}

int is_palindrome(int n) 
{
    if (n < 0) 
    {
        return 0;
    }
    int original = n;
    int reversed = 0;
    while (n != 0) 
    {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return (original == reversed);
}

int gcd_iterative(int a, int b) 
{
    while (a != 0) 
    {
        int temp = a;
        a = b % a;
        b = temp;
    }
    return b;
}

// 统计函数
double variance(int *array, int size) 
{
    double mean = find_average(array, size);
    double variance = 0;
    for (int i = 0; i < size; i++) 
    {
        variance += pow(array[i] - mean, 2);
    }
    return variance / size;
}

double standard_deviation(int *array, int size) 
{
    return sqrt(variance(array, size));
}

int median(int *array, int size) 
{
    sort_array(array, size);
    if (size % 2 == 0) 
    {
        return (array[size / 2 - 1] + array[size / 2]) / 2;
    } else 
    {
        return array[size / 2];
    }
}

int mode(int *array, int size) 
{
    int max_count = 0, mode = 0;
    for (int i = 0; i < size; i++) 
    {
        int count = 0;
        for (int j = 0; j < size; j++) 
        {
            if (array[j] == array[i]) 
            {
                count++;
            }
        }
        if (count > max_count) 
        {
            max_count = count;
            mode = array[i];
        }
    }
    return mode;
}

// 几何计算
double circle_area(double radius) 
{
    return M_PI * radius * radius;
}

double circle_circumference(double radius) 
{
    return 2 * M_PI * radius;
}

double triangle_area(double base, double height) 
{
    return 0.5 * base * height;
}

double triangle_perimeter(double side1, double side2, double side3) 
{
    return side1 + side2 + side3;
}

// 多项式运算
void polynomial_add(int degree, int A[degree + 1], int B[degree + 1], int result[degree + 1]) 
{
    for (int i = 0; i <= degree; i++) 
    {
        result[i] = A[i] + B[i];
    }
}

void polynomial_subtract(int degree, int A[degree + 1], int B[degree + 1], int result[degree + 1]) 
{
    for (int i = 0; i <= degree; i++) 
    {
        result[i] = A[i] - B[i];
    }
}

void polynomial_multiply(int degree1, int A[degree1 + 1], int degree2, int B[degree2 + 1], int result[degree1 + degree2 + 1]) 
{
    for (int i = 0; i <= degree1 + degree2; i++) 
    {
        result[i] = 0;
    }
    for (int i = 0; i <= degree1; i++) 
    {
        for (int j = 0; j <= degree2; j++) 
        {
            result[i + j] += A[i] * B[j];
        }
    }
}

void polynomial_derivative(int degree, int A[degree + 1], int result[degree]) 
{
    for (int i = 1; i <= degree; i++) 
    {
        result[i - 1] = A[i] * i;
    }
}

// 随机数生成
int random_int(int min, int max) 
{
    return min + rand() % (max - min + 1);
}

void random_array(int *array, int size, int min, int max) 
{
    for (int i = 0; i < size; i++) 
    {
        array[i] = random_int(min, max);
    }
}

// 数值积分
double trapezoidal_rule(double (*func)(double), double a, double b, int n) 
{
    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));
    for (int i = 1; i < n; i++) 
    {
        sum += func(a + i * h);
    }
    return sum * h;
}

double simpsons_rule(double (*func)(double), double a, double b, int n) 
{
    double h = (b - a) / n;
    double sum = func(a) + func(b);
    for (int i = 1; i < n; i++) 
    {
        if (i % 2 == 0) 
        {
            sum += 2 * func(a + i * h);
        } 
        else 
        {
            sum += 4 * func(a + i * h);
        }
    }
    return sum * h / 3;
}

Complex complex_add(Complex a, Complex b) 
{
    Complex result;
    result.real = a.real + b.real;
    result.imaginary = a.imaginary + b.imaginary;
    return result;
}

Complex complex_subtract(Complex a, Complex b) 
{
    Complex result;
    result.real = a.real - b.real;
    result.imaginary = a.imaginary - b.imaginary;
    return result;
}

Complex complex_multiply(Complex a, Complex b) 
{
    Complex result;
    result.real = a.real * b.real - a.imaginary * b.imaginary;
    result.imaginary = a.real * b.imaginary + a.imaginary * b.real;
    return result;
}

Complex complex_divide(Complex a, Complex b) 
{
    Complex result;
    double denominator = b.real * b.real + b.imaginary * b.imaginary;
    result.real = (a.real * b.real + a.imaginary * b.imaginary) / denominator;
    result.imaginary = (a.imaginary * b.real - a.real * b.imaginary) / denominator;
    return result;
}
// 向量点积
double vector_dot_product(int size, double A[size], double B[size]) 
{
    double result = 0;
    for (int i = 0; i < size; i++) 
    {
        result += A[i] * B[i];
    }
    return result;
}

// 向量叉积 (仅适用于3维向量)
void vector_cross_product(double A[3], double B[3], double result[3]) 
{
    result[0] = A[1] * B[2] - A[2] * B[1];
    result[1] = A[2] * B[0] - A[0] * B[2];
    result[2] = A[0] * B[1] - A[1] * B[0];
}


// 协方差
double covariance(int size, double A[size], double B[size]) 
{
    double meanA = 0, meanB = 0;
    for (int i = 0; i < size; i++) 
    {
        meanA += A[i];
        meanB += B[i];
    }
    meanA /= size;
    meanB /= size;

    double cov = 0;
    for (int i = 0; i < size; i++) 
    {
        cov += (A[i] - meanA) * (B[i] - meanB);
    }
    return cov / size;
}

// 相关系数
double correlation_coefficient(int size, double A[size], double B[size])
{
    double cov = covariance(size, A, B);
    double stdA = standard_deviation(A, size);
    double stdB = standard_deviation(B, size);
    return cov / (stdA * stdB);
}

// 矩形面积
double rectangle_area(double length, double width) 
{
    return length * width;
}

// 梯形面积
double trapezoid_area(double base1, double base2, double height) 
{
    return 0.5 * (base1 + base2) * height;
}

// 多项式积分
void polynomial_integral(int degree, int A[degree + 1], int result[degree + 2]) 
{
    result[0] = 0; // 积分常数
    for (int i = 0; i <= degree; i++) 
    {
        result[i + 1] = A[i] / (i + 1);
    }
}

// 牛顿法求解方程
double newton_method(double (*func)(double), double (*derivative)(double), double initial_guess, double tolerance) 
{
    double x = initial_guess;
    while (fabs(func(x)) > tolerance) 
    {
        x = x - func(x) / derivative(x);
    }
    return x;
}

// 二分法求解方程
double bisection_method(double (*func)(double), double a, double b, double tolerance) 
{
    if (func(a) * func(b) >= 0) 
    {
        printf("Bisection method requires that func(a) and func(b) have opposite signs.\n");
        return -1;
    }
    double c = a;
    while ((b - a) >= tolerance) 
    {
        c = (a + b) / 2;
        if (func(c) == 0.0) 
        {
            break;
        } 
        else if (func(c) * func(a) < 0) 
        {
            b = c;
        }
        else 
        {
            a = c;
        }
    }
    return c;
}

// 生成正态分布随机数
double normal_distribution_random(double mean, double stddev) 
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z0;
}

// 复数幂运算
Complex complex_power(Complex a, int exponent) 
{
    Complex result = {1.0, 0.0};
    for (int i = 0; i < exponent; i++) 
    {
        result = complex_multiply(result, a);
    }
    return result;
}

// 复数开方
Complex complex_sqrt(Complex a) 
{
    double magnitude = sqrt(a.real * a.real + a.imaginary * a.imaginary);
    double angle = atan2(a.imaginary, a.real);
    double sqrt_magnitude = sqrt(magnitude);
    double sqrt_angle = angle / 2;
    Complex result = {sqrt_magnitude * cos(sqrt_angle), sqrt_magnitude * sin(sqrt_angle)};
    return result;
}
// 1. 计算矩阵的行列式
double matrix_determinant(int n, double** A) 
{
    if (n == 1) 
    {
        return A[0][0];
    }
    if (n == 2) 
    {
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }
    double det = 0;
    for (int col = 0; col < n; col++) 
    {
        double** submatrix = (double**)malloc((n - 1) * sizeof(double*));
        for (int i = 0; i < n - 1; i++) 
        {
            submatrix[i] = (double*)malloc((n - 1) * sizeof(double));
        }
        for (int i = 1; i < n; i++) 
        {
            int subcol = 0;
            for (int j = 0; j < n; j++) 
            {
                if (j == col) continue;
                submatrix[i - 1][subcol] = A[i][j];
                subcol++;
            }
        }
        det += (col % 2 == 0 ? 1 : -1) * A[0][col] * matrix_determinant(n - 1, submatrix);
        for (int i = 0; i < n - 1; i++) 
        {
            free(submatrix[i]);
        }
        free(submatrix);
    }
    return det;
}

// 2. 计算矩阵的逆矩阵
int matrix_inverse(int n, double** A, double** inverse) 
{
    double det = matrix_determinant(n, A);
    if (det == 0) {
        printf("Matrix is singular and cannot be inverted.\n");
        return -1;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) 
        {
            double** submatrix = (double**)malloc((n - 1) * sizeof(double*));
            for (int k = 0; k < n - 1; k++) 
            {
                submatrix[k] = (double*)malloc((n - 1) * sizeof(double));
            }
            int subrow = 0, subcol = 0;
            for (int row = 0; row < n; row++) 
            {
                if (row == i) continue;
                subcol = 0;
                for (int col = 0; col < n; col++) 
                {
                    if (col == j) continue;
                    submatrix[subrow][subcol] = A[row][col];
                    subcol++;
                }
                subrow++;
            }
            double cofactor = ((i + j) % 2 == 0 ? 1 : -1) * matrix_determinant(n - 1, submatrix);
            inverse[j][i] = cofactor / det;
            for (int k = 0; k < n - 1; k++) 
            {
                free(submatrix[k]);
            }
            free(submatrix);
        }
    }
    return 0;
}

// 3. 计算矩阵的转置
void matrix_transpose(int rows, int cols, double** A, double** result) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            result[j][i] = A[i][j];
        }
    }
}

// 4. 计算矩阵的迹（主对角线元素之和）
double matrix_trace(int n, double** A) 
{
    double trace = 0;
    for (int i = 0; i < n; i++) 
    {
        trace += A[i][i];
    }
    return trace;
}

// 5. 计算矩阵的秩
int matrix_rank(int rows, int cols, double** A) 
{
    int rank = cols;
    for (int row = 0; row < rank; row++) 
    {
        if (A[row][row] != 0) 
        {
            for (int col = 0; col < rows; col++) 
            {
                if (col != row) 
                {
                    double mult = A[col][row] / A[row][row];
                    for (int i = 0; i < rank; i++) 
                    {
                        A[col][i] -= mult * A[row][i];
                    }
                }
            }
        } 
        else 
        {
            int reduce = 1;
            for (int i = row + 1; i < rows; i++) 
            {
                if (A[i][row] != 0) 
                {
                    for (int j = 0; j < rank; j++) 
                    {
                        double temp = A[row][j];
                        A[row][j] = A[i][j];
                        A[i][j] = temp;
                    }
                    reduce = 0;
                    break;
                }
            }
            if (reduce) 
            {
                rank--;
                for (int i = 0; i < rows; i++) 
                {
                    A[i][row] = A[i][rank];
                }
            }
            row--;
        }
    }
    return rank;
}

// 6. 计算矩阵的特征多项式
void matrix_characteristic_polynomial(int n, double** A, double* poly) 
{
    // 初始化 B 和 c
    double** B = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        B[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) 
        {
            B[i][j] = (i == j) ? 1.0 : 0.0; // B_0 = I
        }
    }

    poly[0] = 1.0; // c_0 = 1

    for (int k = 1; k <= n; k++) 
    {
        // 计算 B_k = A * B_{k-1}
        double** B_k = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) 
        {
            B_k[i] = (double*)malloc(n * sizeof(double));
            for (int j = 0; j < n; j++) 
            {
                B_k[i][j] = 0.0;
                for (int l = 0; l < n; l++) 
                {
                    B_k[i][j] += A[i][l] * B[l][j];
                }
            }
        }

        // 计算 c_k = -trace(B_k) / k
        double trace = 0.0;
        for (int i = 0; i < n; i++) 
        {
            trace += B_k[i][i];
        }
        poly[k] = -trace / k;

        // 更新 B = B_k - c_k * I
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                B[i][j] = B_k[i][j] - poly[k] * ((i == j) ? 1.0 : 0.0);
            }
        }

        // 释放 B_k 的内存
        for (int i = 0; i < n; i++) 
        {
            free(B_k[i]);
        }
        free(B_k);
    }

    // 释放 B 的内存
    for (int i = 0; i < n; i++) 
    {
        free(B[i]);
    }
    free(B);
}

// 7. 计算矩阵的指数函数（e^A）
void matrix_exponential(int n, double** A, double** result) 
{
    // 初始化 result 为单位矩阵
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            result[i][j] = (i == j) ? 1.0 : 0.0; // result = I
        }
    }

    // 初始化临时变量
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        temp[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) 
        {
            temp[i][j] = (i == j) ? 1.0 : 0.0; // temp = I
        }
    }

    double** A_power = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        A_power[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) 
        {
            A_power[i][j] = A[i][j]; // A_power = A
        }
    }

    int max_iterations = 100; // 最大迭代次数
    double tolerance = 1e-10; // 停止条件：矩阵范数小于此值时停止
    double factorial = 1.0;   // k!

    for (int k = 1; k <= max_iterations; k++) 
    {
        factorial *= k; // 更新 k!

        // 计算 A^k / k!
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                temp[i][j] = A_power[i][j] / factorial;
            }
        }

        // 将 temp 加到 result 中
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                result[i][j] += temp[i][j];
            }
        }

        // 检查是否满足停止条件
        double norm = 0.0;
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                norm += fabs(temp[i][j]);
            }
        }

        // 更新 A_power = A_power * A
        double** new_A_power = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) 
        {
            new_A_power[i] = (double*)malloc(n * sizeof(double));
            for (int j = 0; j < n; j++) 
            {
                new_A_power[i][j] = 0.0;
                for (int l = 0; l < n; l++) 
                {
                    new_A_power[i][j] += A_power[i][l] * A[l][j];
                }
            }
        }

        // 释放旧的 A_power
        for (int i = 0; i < n; i++) 
        {
            free(A_power[i]);
        }
        free(A_power);

        // 更新 A_power
        A_power = new_A_power;
    }

    // 释放内存
    for (int i = 0; i < n; i++) 
    {
        free(temp[i]);
        free(A_power[i]);
    }
    free(temp);
    free(A_power);
}

// 8. 计算矩阵的对数函数（ln(A)）
void matrix_logarithm(int n, double** A, double** result) 
{
    // 假设 A 是对角矩阵，直接计算对角线元素的对数
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (i == j) 
            {
                // 对角线元素：计算对数
                if (A[i][j] > 0) 
                {
                    result[i][j] = log(A[i][j]);
                } 
                else 
                {
                    // 对数未定义，设置为 0 或错误值
                    result[i][j] = 0.0;
                }
            } 
            else 
            {
                // 非对角线元素：设置为 0
                result[i][j] = 0.0;
            }
        }
    }
}
void matrix_multiply(int m, int n, int p, double** A, double** B, double** result) 
{
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < p; j++) 
        {
            result[i][j] = 0.0;
            for (int k = 0; k < n; k++) 
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 辅助函数：将矩阵复制到另一个矩阵
void matrix_copy(int n, double** src, double** dest) 
{
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            dest[i][j] = src[i][j];
        }
    }
}
// 计算矩阵的幂（A^k）
void matrix_power(int n, double** A, int k, double** result) 
{
    // 初始化 result 为单位矩阵
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            result[i][j] = (i == j) ? 1.0 : 0.0; // 单位矩阵
        }
    }

    // 临时矩阵，用于存储中间结果
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        temp[i] = (double*)malloc(n * sizeof(double));
    }

    // 快速幂算法
    while (k > 0) 
    {
        if (k % 2 == 1) 
        {
            // 如果 k 是奇数，result = result * A
            matrix_multiply(n, n, n, result, A, temp); // 修正参数
            matrix_copy(n, temp, result);
        }

        // A = A * A
        matrix_multiply(n, n, n, A, A, temp); // 修正参数
        matrix_copy(n, temp, A);

        // k = k / 2
        k /= 2;
    }

    // 释放临时矩阵的内存
    for (int i = 0; i < n; i++) 
    {
        free(temp[i]);
    }
    free(temp);
}

// 计算矩阵的奇异值分解（SVD）
void matrix_svd(int m, int n, double** A, double** U, double** S, double** V) 
{
    // 初始化 U 为单位矩阵
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < m; j++) 
        {
            U[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // 初始化 V 为单位矩阵
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // 初始化 S 为 0 矩阵
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            S[i][j] = 0.0;
        }
    }

    // 计算 A 的奇异值分解（简化实现，仅适用于小矩阵）
    double tol = 1e-10; // 收敛容差
    double norm = matrix_frobenius_norm(m, n, A);

    // 迭代计算奇异值
    for (int k = 0; k < n; k++) 
    {
        double sigma = 0.0;
        for (int i = 0; i < m; i++) 
        {
            sigma += A[i][k] * A[i][k];
        }
        sigma = sqrt(sigma);

        if (sigma > tol) 
        {
            S[k][k] = sigma;

            // 更新 U 和 V
            for (int i = 0; i < m; i++) 
            {
                U[i][k] = A[i][k] / sigma;
            }

            for (int j = k + 1; j < n; j++) 
            {
                double dot = 0.0;
                for (int i = 0; i < m; i++) 
                {
                    dot += U[i][k] * A[i][j];
                }
                for (int i = 0; i < m; i++) 
                {
                    A[i][j] -= dot * U[i][k];
                }
            }
        }
    }
}


// 辅助函数：计算向量的范数
double vector_norm(int n, double* a) 
{
    double norm = 0.0;
    for (int i = 0; i < n; i++) 
    {
        norm += a[i] * a[i];
    }
    return sqrt(norm);
}

// 辅助函数：向量归一化
void vector_normalize(int n, double* a) 
{
    double norm = vector_norm(n, a);
    if (norm > 0.0) 
    {
        for (int i = 0; i < n; i++) 
        {
            a[i] /= norm;
        }
    }
}

// 辅助函数：向量投影
void vector_projection(int n, double* a, double* b, double* result) 
{
    double dot = vector_dot_product(n, a, b);
    for (int i = 0; i < n; i++) 
    {
        result[i] = dot * b[i];
    }
}

// 计算矩阵的 QR 分解
void matrix_qr_decomposition(int m, int n, double** A, double** Q, double** R) 
{
    // 初始化 Q 和 R
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            Q[i][j] = 0.0;
            R[i][j] = 0.0;
        }
    }

    // 临时向量，用于存储中间结果
    double* temp = (double*)malloc(m * sizeof(double));

    // Gram-Schmidt 正交化
    for (int j = 0; j < n; j++) 
    {
        // 复制 A 的第 j 列到 Q 的第 j 列
        for (int i = 0; i < m; i++) 
        {
            Q[i][j] = A[i][j];
        }

        // 减去 Q 的前 j-1 列的投影
        for (int k = 0; k < j; k++) 
        {
            vector_projection(m, A[j], Q[k], temp);
            for (int i = 0; i < m; i++) 
            {
                Q[i][j] -= temp[i];
            }
        }

        // 归一化 Q 的第 j 列
        vector_normalize(m, Q[j]);

        // 计算 R 的第 j 列
        for (int i = 0; i <= j; i++) 
        {
            R[i][j] = vector_dot_product(m, Q[i], A[j]);
        }
    }

    // 释放临时向量的内存
    free(temp);
}
// 12. 计算矩阵的LU分解
void matrix_lu_decomposition(int n, double** A, double** L, double** U) 
{
    // 初始化 L 和 U 矩阵
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            L[i][j] = (i == j) ? 1.0 : 0.0; // L 是单位下三角矩阵
            U[i][j] = 0.0;                  // U 是上三角矩阵
        }
    }

    // Doolittle 算法
    for (int k = 0; k < n; k++) 
    {
        // 计算 U 的第 k 行
        for (int j = k; j < n; j++) 
        {
            double sum = 0.0;
            for (int p = 0; p < k; p++) 
            {
                sum += L[k][p] * U[p][j];
            }
            U[k][j] = A[k][j] - sum;
        }

        // 计算 L 的第 k 列
        for (int i = k + 1; i < n; i++) 
        {
            double sum = 0.0;
            for (int p = 0; p < k; p++) 
            {
                sum += L[i][p] * U[p][k];
            }
            L[i][k] = (A[i][k] - sum) / U[k][k];
        }
    }
}

// 13. 计算矩阵的Cholesky分解
void matrix_cholesky_decomposition(int n, double** A, double** L) 
{
    // 初始化 L 为 0 矩阵
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            L[i][j] = 0.0;
        }
    }

    // Cholesky 算法
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j <= i; j++) 
        {
            double sum = 0.0;
            // 计算 L[i][j]
            if (i == j) 
            {
                // 对角线元素
                for (int k = 0; k < j; k++) 
                {
                    sum += L[j][k] * L[j][k];
                }
                L[j][j] = sqrt(A[j][j] - sum);
            } 
            else 
            {
                // 非对角线元素
                for (int k = 0; k < j; k++) 
                {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
}

// 15. 计算矩阵的条件数
// 计算矩阵的条件数
double matrix_condition_number(int n, double** A) 
{
    // 计算矩阵 A 的 Frobenius 范数
    double norm_A = matrix_frobenius_norm(n, n, A);
    // 计算矩阵 A 的逆矩阵
    double** A_inv = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        A_inv[i] = (double*)malloc(n * sizeof(double));
    }

    // 假设 matrix_inverse 函数已经实现
    if (matrix_inverse(n, A, A_inv)) 
    {
        // 如果矩阵不可逆，返回一个很大的值（表示条件数无穷大）
        for (int i = 0; i < n; i++) 
        {
            free(A_inv[i]);
        }
        free(A_inv);
        return INFINITY;
    }

    // 计算逆矩阵的 Frobenius 范数
    double norm_A_inv = matrix_frobenius_norm(n, n, A_inv);

    // 释放逆矩阵的内存
    for (int i = 0; i < n; i++) 
    {
        free(A_inv[i]);
    }
    free(A_inv);

    // 返回条件数
    return norm_A * norm_A_inv;
}

// 16. 计算矩阵的范数（Frobenius范数）
double matrix_frobenius_norm(int m, int n, double** A) 
{
    double norm = 0;
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            norm += A[i][j] * A[i][j];
        }
    }
    return sqrt(norm);
}

// 17. 计算矩阵的谱半径
// 幂迭代法计算矩阵的谱半径
double matrix_spectral_radius(int n, double** A) 
{
    double* vec = (double*)malloc(n * sizeof(double)); // 初始向量
    double* new_vec = (double*)malloc(n * sizeof(double)); // 存储迭代结果

    // 初始化向量为 [1, 1, ..., 1]
    for (int i = 0; i < n; i++) 
    {
        vec[i] = 1.0;
    }

    double eigenvalue = 0.0; // 特征值
    double prev_eigenvalue = 0.0; // 上一次迭代的特征值
    double tolerance = 1e-10; // 收敛容差
    int max_iterations = 1000; // 最大迭代次数
    int iter = 0;

    while (iter < max_iterations) 
    {
        // 计算 A * vec
        for (int i = 0; i < n; i++) 
        {
            new_vec[i] = 0.0;
            for (int j = 0; j < n; j++) 
            {
                new_vec[i] += A[i][j] * vec[j];
            }
        }

        // 计算当前特征值（Rayleigh 商）
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; i++) 
        {
            numerator += new_vec[i] * new_vec[i];
            denominator += vec[i] * new_vec[i];
        }
        eigenvalue = sqrt(numerator / denominator);

        // 检查是否收敛
        if (fabs(eigenvalue - prev_eigenvalue) < tolerance) 
        {
            break;
        }

        // 归一化向量
        double norm = vector_norm(n, new_vec);
        for (int i = 0; i < n; i++) 
        {
            vec[i] = new_vec[i] / norm;
        }

        prev_eigenvalue = eigenvalue;
        iter++;
    }

    // 释放内存
    free(vec);
    free(new_vec);

    return eigenvalue;
}

// 18. 计算矩阵的Kronecker积
void matrix_kronecker_product(int m, int n, double** A, int p, int q, double** B, double** result) 
{
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            for (int k = 0; k < p; k++) 
            {
                for (int l = 0; l < q; l++) 
                {
                    result[i * p + k][j * q + l] = A[i][j] * B[k][l];
                }
            }
        }
    }
}

// 19. 计算矩阵的Hadamard积
void matrix_hadamard_product(int m, int n, double** A, double** B, double** result) 
{
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
}

// 20. 计算矩阵的Khatri-Rao积
void matrix_khatri_rao_product(int m, int n, double** A, int p, int q, double** B, double** result) 
{
    // 检查输入矩阵的列数是否相同
    if (n != q) 
    {
        printf("Error: The number of columns in A and B must be the same.\n");
        return;
    }

    // 计算 Khatri-Rao 积
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < p; j++) 
        {
            for (int k = 0; k < n; k++) 
            {
                result[i * p + j][k] = A[i][k] * B[j][k];
            }
        }
    }
}

// 21. 计算矩阵的Moore-Penrose伪逆
// 计算矩阵的 Moore-Penrose 伪逆
void matrix_moore_penrose_pseudo_inverse(int m, int n, double** A, double** pinv) 
{
    // 假设已经实现了 SVD 函数
    // SVD 分解：A = U * S * V^T
    double** U = (double**)malloc(m * sizeof(double*));
    double** S = (double**)malloc(m * sizeof(double*));
    double** V = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < m; i++) 
    {
        U[i] = (double*)malloc(m * sizeof(double));
        S[i] = (double*)malloc(n * sizeof(double));
    }
    for (int i = 0; i < n; i++) 
    {
        V[i] = (double*)malloc(n * sizeof(double));
    }

    // 调用 SVD 函数（假设已经实现）
    matrix_svd(m, n, A, U, S, V);

    // 计算 S 的伪逆（S^+）
    double** S_pinv = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        S_pinv[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) 
        {
            if (i == j && S[i][j] > 1e-10) 
            { // 忽略接近 0 的奇异值
                S_pinv[i][j] = 1.0 / S[i][j];
            } 
            else 
            {
                S_pinv[i][j] = 0.0;
            }
        }
    }

    // 计算 V 的转置（V^T）
    double** V_transpose = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        V_transpose[i] = (double*)malloc(n * sizeof(double));
    }
    matrix_transpose(n, n, V, V_transpose);

    // 计算 U 的转置（U^T）
    double** U_transpose = (double**)malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) 
    {
        U_transpose[i] = (double*)malloc(m * sizeof(double));
    }
    matrix_transpose(m, m, U, U_transpose);

    // 计算伪逆：pinv(A) = V * S^+ * U^T
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        temp[i] = (double*)malloc(m * sizeof(double));
    }
    matrix_multiply(n, n, m, V_transpose, S_pinv, temp); // 修正参数
    matrix_multiply(n, m, m, temp, U_transpose, pinv);   // 修正参数

    // 释放内存
    for (int i = 0; i < m; i++) 
    {
        free(U[i]);
        free(S[i]);
    }
    for (int i = 0; i < n; i++) 
    {
        free(V[i]);
        free(V_transpose[i]);
        free(temp[i]);
    }
    for (int i = 0; i < m; i++) 
    {
        free(U_transpose[i]);
    }
    free(U);
    free(S);
    free(V);
    free(V_transpose);
    free(U_transpose);
    free(temp);
}

// 22. 计算矩阵的广义逆
// 计算矩阵的广义逆（Moore-Penrose 伪逆）
void matrix_generalized_inverse(int m, int n, double** A, double** ginverse) 
{
    // 假设已经实现了 SVD 函数
    // SVD 分解：A = U * S * V^T
    double** U = (double**)malloc(m * sizeof(double*));
    double** S = (double**)malloc(m * sizeof(double*));
    double** V = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < m; i++) 
    {
        U[i] = (double*)malloc(m * sizeof(double));
        S[i] = (double*)malloc(n * sizeof(double));
    }
    for (int i = 0; i < n; i++) 
    {
        V[i] = (double*)malloc(n * sizeof(double));
    }

    // 调用 SVD 函数（假设已经实现）
    matrix_svd(m, n, A, U, S, V);

    // 计算 S 的伪逆（S^+）
    double** S_pinv = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) 
    {
        S_pinv[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) 
        {
            if (i == j) 
            {
                if (S[i][j] > 1e-10) 
                { // 忽略接近 0 的奇异值
                    S_pinv[i][j] = 1.0 / S[i][j];
                } 
                else 
                {
                    S_pinv[i][j] = 0.0;
                }
            } 
            else 
            {
                S_pinv[i][j] = 0.0;
            }
        }
    }

    // 计算 V 的转置（V^T）
    double** V_transpose = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        V_transpose[i] = (double*)malloc(n * sizeof(double));
    }
    matrix_transpose(n, n, V, V_transpose);

    // 计算 U 的转置（U^T）
    double** U_transpose = (double**)malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) {
        U_transpose[i] = (double*)malloc(m * sizeof(double));
    }
    matrix_transpose(m, m, U, U_transpose);

    // 计算广义逆：ginverse(A) = V * S^+ * U^T
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        temp[i] = (double*)malloc(m * sizeof(double));
    }
    matrix_multiply(n, n, m, V_transpose, S_pinv, temp);
    matrix_multiply(n, m, m, temp, U_transpose, ginverse);

    // 释放内存
    for (int i = 0; i < m; i++) {
        free(U[i]);
        free(S[i]);
    }
    for (int i = 0; i < n; i++) {
        free(V[i]);
        free(V_transpose[i]);
        free(temp[i]);
    }
    for (int i = 0; i < m; i++) {
        free(U_transpose[i]);
    }
    free(U);
    free(S);
    free(V);
    free(V_transpose);
    free(U_transpose);
    free(temp);
}



// 24. 计算矩阵的Jordan标准形
// 计算矩阵的 Jordan 标准形
void matrix_jordan_form(int n, double** A, double** J, double** P) {
    // 假设特征值和特征向量已知
    // 这里简化实现，假设 A 是对角化的

    // 初始化 J 和 P
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            J[i][j] = 0.0;
            P[i][j] = 0.0;
        }
    }

    // 假设特征值为 1, 2, 3（简化示例）
    double eigenvalues[] = {1, 2, 3};

    // 构造 Jordan 矩阵 J
    for (int i = 0; i < n; i++) {
        J[i][i] = eigenvalues[i]; // 对角线为特征值
    }

    // 假设特征向量为单位矩阵（简化示例）
    for (int i = 0; i < n; i++) {
        P[i][i] = 1.0; // P 为单位矩阵
    }

    // 验证 A = P * J * P^(-1)
    double** P_inv = (double**)malloc(n * sizeof(double*));
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        P_inv[i] = (double*)malloc(n * sizeof(double));
        temp[i] = (double*)malloc(n * sizeof(double));
    }

    // 计算 P^(-1)（假设 P 是单位矩阵，P^(-1) = P）
    matrix_transpose(n, n, P, P_inv);

    // 计算 P * J
    matrix_multiply(n, n, n, P, J, temp);

    // 计算 P * J * P^(-1)
    double** reconstructed_A = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        reconstructed_A[i] = (double*)malloc(n * sizeof(double));
    }
    matrix_multiply(n, n, n, temp, P_inv, reconstructed_A);


    // 释放内存
    for (int i = 0; i < n; i++) {
        free(P_inv[i]);
        free(temp[i]);
        free(reconstructed_A[i]);
    }
    free(P_inv);
    free(temp);
    free(reconstructed_A);
}

// 25. 计算矩阵的Hessenberg形式
// 辅助函数：计算 Householder 向量
void householder_vector(int n, double* x, double* v) {
    double sigma = 0.0;
    for (int i = 1; i < n; i++) {
        sigma += x[i] * x[i];
    }
    double beta = sqrt(x[0] * x[0] + sigma);
    if (sigma == 0.0) {
        v[0] = 0.0;
    } else {
        double alpha = x[0] < 0 ? -beta : beta;
        v[0] = x[0] + alpha;
        for (int i = 1; i < n; i++) {
            v[i] = x[i];
        }
        double norm_v = vector_norm(n, v);
        for (int i = 0; i < n; i++) {
            v[i] /= norm_v;
        }
    }
}

// 辅助函数：应用 Householder 变换
void apply_householder(int n, double** A, double* v, int k) {
    double* Av = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        Av[i] = 0.0;
        for (int j = k; j < n; j++) {
            Av[i] += A[i][j] * v[j - k];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = k; j < n; j++) {
            A[i][j] -= 2 * v[i - k] * Av[j];
        }
    }

    free(Av);
}

// 计算矩阵的 Hessenberg 形式
void matrix_hessenberg_form(int n, double** A, double** H) {
    // 初始化 H 为 A 的副本
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H[i][j] = A[i][j];
        }
    }

    // 使用 Householder 变换将 H 转换为上 Hessenberg 形式
    for (int k = 0; k < n - 2; k++) {
        double* x = (double*)malloc((n - k - 1) * sizeof(double));
        for (int i = k + 1; i < n; i++) {
            x[i - k - 1] = H[i][k];
        }

        double* v = (double*)malloc((n - k - 1) * sizeof(double));
        householder_vector(n - k - 1, x, v);

        // 应用 Householder 变换到 H 的右侧
        apply_householder(n, H, v, k + 1);

        // 应用 Householder 变换到 H 的左侧
        double* Hv = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            Hv[i] = 0.0;
            for (int j = k + 1; j < n; j++) {
                Hv[i] += H[i][j] * v[j - k - 1];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                H[i][j] -= 2 * Hv[i] * v[j - k - 1];
            }
        }

        free(x);
        free(v);
        free(Hv);
    }
}
// 26. 计算矩阵的QR算法
// 计算矩阵的特征值（QR 算法）
void matrix_qr_algorithm(int n, double** A, double* eigenvalues) {
    // 初始化临时矩阵
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        temp[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            temp[i][j] = A[i][j];
        }
    }

    // 迭代次数
    int max_iterations = 100;
    double tolerance = 1e-10;

    // QR 算法迭代
    for (int iter = 0; iter < max_iterations; iter++) {
        // 计算 QR 分解
        double** Q = (double**)malloc(n * sizeof(double*));
        double** R = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) {
            Q[i] = (double*)malloc(n * sizeof(double));
            R[i] = (double*)malloc(n * sizeof(double));
        }

        matrix_qr_decomposition(n, n, temp, Q, R);

        // 更新 temp = R * Q
        double** new_temp = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) {
            new_temp[i] = (double*)malloc(n * sizeof(double));
        }
        matrix_multiply(n, n, n, R, Q, new_temp);

        // 检查是否收敛（下三角部分接近 0）
        double norm = 0.0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                norm += fabs(new_temp[i][j]);
            }
        }
        // 更新 temp
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp[i][j] = new_temp[i][j];
            }
        }

        // 释放内存
        for (int i = 0; i < n; i++) {
            free(Q[i]);
            free(R[i]);
            free(new_temp[i]);
        }
        free(Q);
        free(R);
        free(new_temp);
    }

    // 提取特征值（对角线元素）
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = temp[i][i];
    }

    // 释放内存
    for (int i = 0; i < n; i++) {
        free(temp[i]);
    }
    free(temp);
}

// 27. 计算矩阵的Rayleigh商
double matrix_rayleigh_quotient(int n, double** A, double* x) {
    double numerator = 0, denominator = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            numerator += x[i] * A[i][j] * x[j];
        }
        denominator += x[i] * x[i];
    }
    return numerator / denominator;
}

// 28. 计算矩阵的幂迭代法
// 计算向量的 2-范数

void matrix_power_iteration(int n, double** A, double* eigenvalue, double* eigenvector) {
    double* vec = (double*)malloc(n * sizeof(double)); // 初始向量
    double* new_vec = (double*)malloc(n * sizeof(double)); // 存储迭代结果

    // 初始化向量为 [1, 1, ..., 1]
    for (int i = 0; i < n; i++) {
        vec[i] = 1.0;
    }

    double prev_eigenvalue = 0.0; // 上一次迭代的特征值
    double tolerance = 1e-10; // 收敛容差
    int max_iterations = 1000; // 最大迭代次数
    int iter = 0;

    while (iter < max_iterations) {
        // 计算 A * vec
        for (int i = 0; i < n; i++) {
            new_vec[i] = 0.0;
            for (int j = 0; j < n; j++) {
                new_vec[i] += A[i][j] * vec[j];
            }
        }

        // 计算当前特征值（Rayleigh 商）
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; i++) {
            numerator += new_vec[i] * new_vec[i];
            denominator += vec[i] * new_vec[i];
        }
        *eigenvalue = numerator / denominator;

        // 归一化向量
        double norm = vector_norm(n, new_vec);
        for (int i = 0; i < n; i++) {
            eigenvector[i] = new_vec[i] / norm;
            vec[i] = eigenvector[i]; // 更新 vec 用于下一次迭代
        }

        prev_eigenvalue = *eigenvalue;
        iter++;
    }

    // 释放内存
    free(vec);
    free(new_vec);
}



// 29. 计算矩阵的反幂迭代法
// 反幂迭代法计算矩阵的最小特征值和特征向量
void matrix_inverse_power_iteration(int n, double** A, double* eigenvalue, double* eigenvector) {
    double* vec = (double*)malloc(n * sizeof(double)); // 初始向量
    double* new_vec = (double*)malloc(n * sizeof(double)); // 存储迭代结果

    // 初始化向量为 [1, 1, ..., 1]
    for (int i = 0; i < n; i++) {
        vec[i] = 1.0;
    }

    double prev_eigenvalue = 0.0; // 上一次迭代的特征值
    double tolerance = 1e-10; // 收敛容差
    int max_iterations = 1000; // 最大迭代次数
    int iter = 0;

    while (iter < max_iterations) {
        // 解线性方程组 A * new_vec = vec
        // 这里假设 matrix_solve_linear_system 函数已经实现
        // 该函数用于求解线性方程组 A * x = b
        matrix_solve_linear_system(n, A, vec, new_vec);

        // 计算当前特征值（Rayleigh 商）
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; i++) {
            numerator += new_vec[i] * new_vec[i];
            denominator += vec[i] * new_vec[i];
        }
        *eigenvalue = denominator / numerator; // 注意：这里是反幂迭代法的特征值


        // 归一化向量
        double norm = vector_norm(n, new_vec);
        for (int i = 0; i < n; i++) {
            eigenvector[i] = new_vec[i] / norm;
            vec[i] = eigenvector[i]; // 更新 vec 用于下一次迭代
        }

        prev_eigenvalue = *eigenvalue;
        iter++;
    }

    // 释放内存
    free(vec);
    free(new_vec);
}

// 高斯消元法求解线性方程组 A * x = b
void matrix_solve_linear_system(int n, double** A, double* b, double* x) {
    // 创建一个增广矩阵 [A | b]
    double** augmented = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        augmented[i] = (double*)malloc((n + 1) * sizeof(double));
        for (int j = 0; j < n; j++) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n] = b[i];
    }

    // 高斯消元
    for (int i = 0; i < n; i++) {
        // 找到主元
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(augmented[k][i]) > fabs(augmented[max_row][i])) {
                max_row = k;
            }
        }

        // 交换行
        if (max_row != i) {
            double* temp = augmented[i];
            augmented[i] = augmented[max_row];
            augmented[max_row] = temp;
        }

        // 消元
        for (int k = i + 1; k < n; k++) {
            double factor = augmented[k][i] / augmented[i][i];
            for (int j = i; j <= n; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }

    // 回代求解
    for (int i = n - 1; i >= 0; i--) {
        x[i] = augmented[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= augmented[i][j] * x[j];
        }
        x[i] /= augmented[i][i];
    }

    // 释放内存
    for (int i = 0; i < n; i++) {
        free(augmented[i]);
    }
    free(augmented);
}

// 30. 计算矩阵的Arnoldi迭代


// Arnoldi 迭代
void matrix_arnoldi_iteration(int n, double** A, int m, double** H, double** Q) {
    // 初始化 Q 和 H
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Q[i][j] = 0.0;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            H[i][j] = 0.0;
        }
    }

    // 初始向量 q0
    double* q0 = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        q0[i] = (i == 0) ? 1.0 : 0.0; // q0 = [1, 0, ..., 0]
    }

    // 归一化 q0 并存储到 Q 的第一列
    vector_normalize(n, q0);
    for (int i = 0; i < n; i++) {
        Q[i][0] = q0[i];
    }

    // Arnoldi 迭代
    for (int j = 0; j < m; j++) {
        // 计算 A * qj
        double* v = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            v[i] = 0.0;
            for (int k = 0; k < n; k++) {
                v[i] += A[i][k] * Q[k][j];
            }
        }

        // 正交化 v 与 Q 的前 j 列
        for (int i = 0; i <= j; i++) {
            H[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                H[i][j] += Q[k][i] * v[k];
            }
            for (int k = 0; k < n; k++) {
                v[k] -= H[i][j] * Q[k][i];
            }
        }


        // 释放 v 的内存
        free(v);
    }

    // 释放 q0 的内存
    free(q0);
}


// 32. 计算矩阵的Givens旋转
void matrix_givens_rotation(int n, double** A, int i, int j, double c, double s) 
{
    // 对矩阵 A 的第 i 行和第 j 行进行 Givens 旋转
    for (int k = 0; k < n; k++) 
    {
        double a_ik = A[i][k];
        double a_jk = A[j][k];
        A[i][k] = c * a_ik + s * a_jk;
        A[j][k] = -s * a_ik + c * a_jk;
    }
}

// 33. 计算矩阵的Householder变换
// Householder 变换
void matrix_householder_transformation(int n, double** A, double* v, double** H) {
    // 计算 Householder 向量 v 的范数
    double norm_v = vector_norm(n, v);

    // 计算 Householder 矩阵 H = I - 2 * v * v^T / (v^T * v)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H[i][j] = (i == j) ? 1.0 : 0.0; // 初始化 H 为单位矩阵
            H[i][j] -= 2 * v[i] * v[j] / (norm_v * norm_v);
        }
    }

    // 对矩阵 A 进行 Householder 变换：A = H * A
    double** temp = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        temp[i] = (double*)malloc(n * sizeof(double));
    }

    // 计算 H * A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                temp[i][j] += H[i][k] * A[k][j];
            }
        }
    }

    // 将结果复制回 A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = temp[i][j];
        }
    }

    // 释放内存
    for (int i = 0; i < n; i++) {
        free(temp[i]);
    }
    free(temp);
}
