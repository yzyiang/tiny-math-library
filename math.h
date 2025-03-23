#ifndef MATH_H
#define MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// 基本数学运算
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
int divide(int a, int b);

// 阶乘和斐波那契数列
int factorial(int n);
int fibonacci(int n);

// 素数判断
int is_prime(int n);

// 最大公约数和最小公倍数
int gcd(int a, int b);
int lcm(int a, int b);

// 幂运算和取模运算
int power(int base, int exponent);
int modulo(int a, int b);

// 数组操作
int sum_array(int *array, int size);
int product_array(int *array, int size);
int find_max(int *array, int size);
int find_min(int *array, int size);
int find_average(int *array, int size);
int sort_array(int *array, int size);

// 搜索算法
int binary_search(int *array, int size, int value);
int linear_search(int *array, int size, int value);

// 新增函数
int absolute(int n);                     // 计算绝对值
int sqrt_int(int n);                     // 计算整数的平方根（返回整数部分）
int is_perfect_square(int n);            // 判断一个数是否是完全平方数
float harmonic_mean(int a, int b);       // 计算两个数的调和平均数
int cube(int n);                         // 计算一个数的立方
int cube_root(int n);                    // 计算一个数的立方根（返回整数部分）
int sum_of_factorials(int n);            // 计算从 1 到 n 的所有整数的阶乘的和
int count_digits(int n);                 // 计算一个整数的位数
int is_palindrome(int n);                // 判断一个数是否为回文数
int gcd_iterative(int a, int b);         // 非递归版本的 GCD 计算

// 统计函数
double variance(int *array, int size);
double standard_deviation(int *array, int size);
int median(int *array, int size);
int mode(int *array, int size);


// 几何计算
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
double circle_area(double radius);
double circle_circumference(double radius);
double triangle_area(double base, double height);
double triangle_perimeter(double side1, double side2, double side3);

// 多项式运算
void polynomial_add(int degree, int A[degree + 1], int B[degree + 1], int result[degree + 1]);
void polynomial_subtract(int degree, int A[degree + 1], int B[degree + 1], int result[degree + 1]);
void polynomial_multiply(int degree1, int A[degree1 + 1], int degree2, int B[degree2 + 1], int result[degree1 + degree2 + 1]);
void polynomial_derivative(int degree, int A[degree + 1], int result[degree]);

// 随机数生成
int random_int(int min, int max);
void random_array(int *array, int size, int min, int max);

// 数值积分
double trapezoidal_rule(double (*func)(double), double a, double b, int n);
double simpsons_rule(double (*func)(double), double a, double b, int n);

// 线性代数
void solve_linear_system(int n, double A[n][n], double B[n], double X[n]);
void eigenvalues(int n, double A[n][n], double eigenvalues[n]);
void eigenvectors(int n, double A[n][n], double eigenvectors[n][n]);

// 复数运算
typedef struct {
    double real;
    double imaginary;
} Complex;

Complex complex_add(Complex a, Complex b);
Complex complex_subtract(Complex a, Complex b);
Complex complex_multiply(Complex a, Complex b);
Complex complex_divide(Complex a, Complex b);

// 向量运算
double vector_dot_product(int size, double A[size], double B[size]);
void vector_cross_product(double A[3], double B[3], double result[3]);
double vector_magnitude(int size, double A[size]);

// 统计函数
double covariance(int size, double A[size], double B[size]);
double correlation_coefficient(int size, double A[size], double B[size]);

// 几何计算
double rectangle_area(double length, double width);
double rectangle_perimeter(double length, double width);
double trapezoid_area(double base1, double base2, double height);
double trapezoid_perimeter(double base1, double base2, double side1, double side2);

// 多项式运算
void polynomial_integral(int degree, int A[degree + 1], int result[degree + 2]);

// 数值分析
double newton_method(double (*func)(double), double (*derivative)(double), double initial_guess, double tolerance);
double bisection_method(double (*func)(double), double a, double b, double tolerance);

// 随机数生成
double normal_distribution_random(double mean, double stddev);

// 复数运算
Complex complex_power(Complex a, int exponent);
Complex complex_sqrt(Complex a);

// 矩阵运算相关函数声明
void matrix_multiply(int m, int n, int p, double** A, double** B, double** result);
void matrix_copy(int n, double** src, double** dest);
// 计算 Householder 向量
void householder_vector(int n, double* x, double* v);
// 应用 Householder 变换
void apply_householder(int n, double** A, double* v, int k);
void matrix_solve_linear_system(int n, double** A, double* b, double* x);
void vector_normalize(int n, double* a);
double matrix_determinant(int n, double** A);
int matrix_inverse(int n, double** A, double** inverse);
void matrix_transpose(int rows, int cols, double** A, double** result);
double matrix_trace(int n, double** A);
int matrix_rank(int rows, int cols, double** A);
void matrix_characteristic_polynomial(int n, double** A, double* poly);
void matrix_exponential(int n, double** A, double** result);
void matrix_logarithm(int n, double** A, double** result);
void matrix_power(int n, double** A, int k, double** result);
void matrix_svd(int m, int n, double** A, double** U, double** S, double** V);
void matrix_qr_decomposition(int m, int n, double** A, double** Q, double** R);
void matrix_lu_decomposition(int n, double** A, double** L, double** U);
void matrix_cholesky_decomposition(int n, double** A, double** L);
void matrix_pseudo_inverse(int m, int n, double** A, double** pinv);
double matrix_condition_number(int n, double** A);
double matrix_frobenius_norm(int m, int n, double** A);
double matrix_spectral_radius(int n, double** A);
void matrix_kronecker_product(int m, int n, double** A, int p, int q, double** B, double** result);
void matrix_hadamard_product(int m, int n, double** A, double** B, double** result);
void matrix_khatri_rao_product(int m, int n, double** A, int p, int q, double** B, double** result);
void matrix_moore_penrose_pseudo_inverse(int m, int n, double** A, double** pinv);
void matrix_generalized_inverse(int m, int n, double** A, double** ginverse);
void matrix_schur_decomposition(int n, double** A, double** T, double** Q);
void matrix_jordan_form(int n, double** A, double** J, double** P);
void matrix_hessenberg_form(int n, double** A, double** H);
void matrix_qr_algorithm(int n, double** A, double* eigenvalues);
double matrix_rayleigh_quotient(int n, double** A, double* x);
void matrix_power_iteration(int n, double** A, double* eigenvalue, double* eigenvector);
void matrix_inverse_power_iteration(int n, double** A, double* eigenvalue, double* eigenvector);
void matrix_arnoldi_iteration(int n, double** A, int m, double** H, double** Q);
void matrix_lanczos_iteration(int n, double** A, int m, double** T, double** Q);
void matrix_givens_rotation(int n, double** A, int i, int j, double c, double s);
void matrix_householder_transformation(int n, double** A, double* v, double** H);
void matrix_gram_schmidt_orthogonalization(int m, int n, double** A, double** Q);

#ifdef __cplusplus
}
#endif

#endif // MATH_H
