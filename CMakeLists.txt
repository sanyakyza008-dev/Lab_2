cmake_minimum_required(VERSION 3.10)
project(Lab2_SD)

# Флаги для максимальной скорости
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)

# Указываем путь к папке lib
set(LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Если тяжелого dll файла нет, но есть zip-архив, CMake сам его распакует
if(NOT EXISTS "${LOCAL_LIB_DIR}/libopenblas.dll" AND EXISTS "${LOCAL_LIB_DIR}/libopenblas.zip")
    message(STATUS "Extracting libopenblas.zip...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "libopenblas.zip"
                    WORKING_DIRECTORY "${LOCAL_LIB_DIR}")
endif()

# Создание исполнемых файлов
add_executable(MatrixGenerator MatrixGenerator.cpp)
add_executable(calculator calculator.cpp)

# Чтобы компилятор искал cblas.h в папке проекта
target_include_directories(calculator PRIVATE ${LOCAL_LIB_DIR})

# Привязываем локальную библиотеку напрямую
target_link_libraries(calculator PRIVATE "${LOCAL_LIB_DIR}/libopenblas.dll.a" OpenMP::OpenMP_CXX)

# Эта команда автоматически копирует libopenblas.dll к calculator.exe
add_custom_command(TARGET calculator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${LOCAL_LIB_DIR}/libopenblas.dll"
    $<TARGET_FILE_DIR:calculator>
)