cmake_minimum_required(VERSION 3.20)

add_executable(ghostwolf_cli main.cpp)

target_link_libraries(ghostwolf_cli PRIVATE ghostwolf_engine)

target_compile_features(ghostwolf_cli PRIVATE cxx_std_20)