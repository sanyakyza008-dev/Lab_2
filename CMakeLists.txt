cmake_minimum_required(VERSION 3.10)
project(Lab2_SD)

# 1. Максимальная скорость по умолчанию
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)

# 2. Указываем путь к нашей новой папке lib
set(LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

add_executable(MatrixGenerator MatrixGenerator.cpp)
add_executable(calculator calculator.cpp)

# 3. Говорим компилятору искать cblas.h в нашей папке
target_include_directories(calculator PRIVATE ${LOCAL_LIB_DIR})

# 4. Привязываем нашу локальную библиотеку напрямую
target_link_libraries(calculator PRIVATE "${LOCAL_LIB_DIR}/libopenblas.dll.a" OpenMP::OpenMP_CXX)

# 5. МАГИЯ ДЛЯ 100% СОВМЕСТИМОСТИ:
# Эта команда автоматически копирует libopenblas.dll прямо к готовому calculator.exe
# после каждой сборки. Преподавателю не придется ничего скачивать.
add_custom_command(TARGET calculator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${LOCAL_LIB_DIR}/libopenblas.dll"
    $<TARGET_FILE_DIR:calculator>
)