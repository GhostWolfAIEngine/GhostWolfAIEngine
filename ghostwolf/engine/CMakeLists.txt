cmake_minimum_required(VERSION 3.20)
project(GhostWolfEngine LANGUAGES CXX)

add_library(ghostwolf_engine
	src/gw_engine.cpp
	src/voxel_grid.cpp
	src/rays.cpp
	src/traversal.cpp
	src/aggregation.cpp
	src/scheduler.cpp
)

target_include_directories(ghostwolf_engine PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
	target_compile_options(ghostwolf_engine PRIVATE /W4 /EHsc)
else()
	target_compile_options(ghostwolf_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(ghostwolf_engine PRIVATE $<$<PLATFORM_ID:Windows>:GHOSTWOLF_WINDOWS>)