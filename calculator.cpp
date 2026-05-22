#include <iostream>
#include <vector>
#include <algorithm> // Алгоритмы обработки данных
#include <chrono> // Измерение времени
#include <iomanip> // Форматирование вывода
#include <fstream>
#include <filesystem>
#include <cblas.h> // Интерфейс с библиотеке BLAS
#include <omp.h> // Чтобы параллелить циклы. Технология OpenMP
#include <cmath> // Для std::sqrt и std::abs

#ifdef _WIN32
#include <windows.h>
#endif


namespace fs = std::filesystem;

constexpr int N = 4096;

// Функция для бинарного чтения
bool load_matrix(const fs::path& filepath, std::vector<float>& matrix) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    // reinterpret_cast - оператор приведения типов, не меняющий самого значения данных
    file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(float));
    return true;
}

// Оценка производительности
void print_metrics(const std::string& name, double time_seconds) {
    // Перевод в double, чтобы избежать переполнения типов
    double c = 2.0 * static_cast<double>(N) * N * N;
    double p = (c / time_seconds) * 1e-6;
    // Манипуляторы для форматирования вывода
    std::cout << std::left << std::setw(20) << name 
              << " | Время: " << std::fixed << std::setprecision(4) << time_seconds << " с"
              << " | Производительность: " << std::fixed << std::setprecision(0) << p << " MFlops\n";
}

// 1. Наивный вариант
void multiply_naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. Вариант BLAS
// CblasRowMajor указывает, что матрицы лежат в памяти по строкам (как принято в C++), а не по столбцам
// CblasNoTrans для двух матриц говорит библиотеке, что матрицы A и B не нужно транспонировать перед умножением
// N, N, N - размеры матрицы
// 1.0f это коэффициент альфа, 0.0f - бета
// A.data(), N - указатель на начало данных матрицы A и её шаг. То же самое для В
// C.data(), N - сюда записываем результат
void multiply_blas(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, A.data(), N,
                B.data(), N,
                0.0f, C.data(), N);
}

// 3. Оптимизированный вариант
// Использованы tiling, изменение порядка циклов и параллелизм
void multiply_optimized(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    constexpr int BLOCK_SIZE = 64;
    // В первых трех бью на блоки по 64 Кб для кешей L1/L2
    #pragma omp parallel for collapse(2) // Создает команды потоков и разделяет между ядрами
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE; ++ii) {
                    for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
                        float a_ik = A[ii * N + kk];
                        #pragma omp simd // команда использовать векторные регистры
                        for (int jj = j; jj < j + BLOCK_SIZE; ++jj) {
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

// Функция для оценки точности (Относительная норма Фробениуса)
double check_accuracy(const std::vector<float>& C_ref, const std::vector<float>& C_test) {
    double diff_sq_sum = 0.0;
    double ref_sq_sum = 0.0;

    // Используем OpenMP для быстрого подсчета суммы квадратов
    #pragma omp parallel for reduction(+:diff_sq_sum, ref_sq_sum)
    for (int i = 0; i < N * N; ++i) {
        // Обязательно переводим во float->double ДО вычитания, 
        // чтобы избежать переполнения при суммировании 16 млн элементов!
        double diff = static_cast<double>(C_ref[i]) - static_cast<double>(C_test[i]);
        double ref = static_cast<double>(C_ref[i]);
        
        diff_sq_sum += diff * diff;
        ref_sq_sum += ref * ref;
    }

    if (ref_sq_sum == 0.0) return 0.0;
    return std::sqrt(diff_sq_sum) / std::sqrt(ref_sq_sum);
}

int main(int argc, char* argv[]) {
    #ifdef _WIN32
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    #endif

    // Этот подход гарантирует, что матрицы всегда будут лежать в той же папке, где находится сам .exe файл
    fs::path exe_dir = fs::absolute(argv[0]).parent_path();
    fs::path file_A = exe_dir / "matrix_A.bin";
    fs::path file_B = exe_dir / "matrix_B.bin";

    std::cout << "Загрузка матриц из " << exe_dir.string() << "...\n";

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    if (!load_matrix(file_A, A) || !load_matrix(file_B, B)) {
        std::cerr << "[ОШИБКА] Не удалось найти файлы. Запустите generator\n";
        return 1;
    }
    std::cout << "Загрузка завершена. Произвожу вычисления...\n\n";

    double time_seconds;

    std::cout << "1. Наивный алгоритм\n";
    std::cout << "   [ПРОПУЩЕНО] Сложность 2*n^3. Будет считаться очень долго\n";
    // std::fill(C.begin(), C.end(), 0.0f);
    // auto start = std::chrono::high_resolution_clock::now();
    // multiply_naive(A, B, C);
    // auto end = std::chrono::high_resolution_clock::now();
    // time_seconds = std::chrono::duration<double>(end - start).count();
    // print_metrics("1. Наивный", diff.count());

    std::fill(C.begin(), C.end(), 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    multiply_blas(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    time_seconds = std::chrono::duration<double>(end - start).count();
    print_metrics("2. BLAS", time_seconds);

    std::vector<float> C_blas = C;

    std::fill(C.begin(), C.end(), 0.0f);
    start = std::chrono::high_resolution_clock::now();
    multiply_optimized(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    time_seconds = std::chrono::duration<double>(end - start).count();
    print_metrics("3. Оптимизированный", time_seconds);

    std::cout << "\nПроверка точности вычислений (норма Фробениуса)...\n";
    double error = check_accuracy(C_blas, C);
    
    // Выводим в научном формате (с экспонентой)
    std::cout << "Относительная ошибка: " << std::scientific << error << "\n";

    std::cout << "Работу выполнил: Кузьмич А Д группа 090301-ПОВа-025" << std::endl;
    return 0;
}