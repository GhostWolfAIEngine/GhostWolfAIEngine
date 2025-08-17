cmake_minimum_required(VERSION 3.20)
project(GhostWolf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GHOSTWOLF_BUILD_CLI "Build the GhostWolf CLI" ON)

add_subdirectory(engine)

if(GHOSTWOLF_BUILD_CLI)
	add_subdirectory(apps/cli)
endif()