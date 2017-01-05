CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

add_definitions(-std=c++11 -fPIC)

ADD_LIBRARY(TENSORVIS SHARED visualizeTensor.cpp cMarchingCubes.cpp cProtein.cpp)
target_link_libraries(TENSORVIS TH TH_MATH GLFRAMEWORK)