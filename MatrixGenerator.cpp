#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif


namespace fs = std::filesystem;

constexpr int N = 4096;

// Функция для бинарного сохранения
bool save_matrix(const fs::path& filepath, const std::vector<float>& matrix) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    file.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(float));
    return true;
}

int main(int argc, char* argv[]) {
    #ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    #endif

    fs::path exe_dir = fs::absolute(argv[0]).parent_path();
    fs::path file_A = exe_dir / "matrix_A.bin";
    fs::path file_B = exe_dir / "matrix_B.bin";

    std::cout << "Рабочая папка: " << exe_dir.string() << "\n";

    if (fs::exists(file_A) && fs::exists(file_B)) {
        std::cout << "[!] Файлы matrix_A.bin и matrix_B.bin уже существуют\n";
        std::cout << "Создание пропущено. Можно запускать calculator\n";
        return 0;
    }

    std::cout << "Файлы не найдены. Выделение памяти для " << N << "x" << N << " матриц...\n";
    
    std::vector<float> A; 
    std::vector<float> B;

    A.reserve(N * N);
    B.reserve(N * N);

    std::cout << "Создание случайных данных...\n";
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < N * N; ++i) {
        A.push_back(dist(gen)); // Передаю функторы
        B.push_back(dist(gen));
    }

    std::cout << "Сохранение в файлы...\n";
    if (save_matrix(file_A, A) && save_matrix(file_B, B)) {
        std::cout << "[УСПЕШНО] Матрицы успешно сохранены\n";
    } else {
        std::cerr << "[ОШИБКА] Не удалось сохранить файлы\n";
    }

    return 0;
}