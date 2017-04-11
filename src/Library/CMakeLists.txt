CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 7.5 REQUIRED)

IF(DEFINED Torch_INSTALL_INCLUDE)
  SET(TH_INSTALL_INCLUDE ${Torch_INSTALL_INCLUDE})
  SET(TH_INSTALL_LIB ${Torch_INSTALL_LIB})
ENDIF()

INCLUDE_DIRECTORIES(
  ${TH_INSTALL_INCLUDE}/TH
  ${TH_INSTALL_INCLUDE}
)
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
#INCLUDE_DIRECTORIES("/home/lupoglaz/NVIDIA_CUDA-7.5_Samples/common/inc")

LINK_DIRECTORIES(${TH_INSTALL_LIB})

add_definitions(-std=c++11 -fPIC)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20 -Xcompiler -fPIC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")



#TH_MATH: tensor scaling, rotating and saving to ccp4
include_directories("Math")
add_subdirectory(Math)

#PROTEINLOADER: protein loading library
include_directories("CppProteinLoader")
add_subdirectory(CppProteinLoader)

#GLFRAMEWORK: freeglut visualization library
#include_directories("GL")
#add_subdirectory(GL)

#TENSORVIS: tensor visualization library
#include_directories("VolumeVisualization")
#add_subdirectory(VolumeVisualization)




# library for loading protein structure from torch
ADD_LIBRARY(load_protein SHARED CppProteinLoader/loadProtein.cpp)
TARGET_LINK_LIBRARIES(load_protein TH PROTEINLOADER)

#Load all proteins in the dataset and check exceptions
#add_executable(buildCorruptedFilesList CppProteinLoader/buildCorruptedFilesList.cpp)
#TARGET_LINK_LIBRARIES(buildCorruptedFilesList TH PROTEINLOADER)

#Test tensor visualization using protein loading and visualizing it's density
#add_executable(marchingCubes VolumeVisualization/main.cpp)
#target_link_libraries (marchingCubes TH GLFRAMEWORK PROTEINLOADER TENSORVIS)

#External function to visualize a tensor from lua
#ADD_LIBRARY(visualizeTensor SHARED VolumeVisualization/visualizeTensor.cpp)
#target_link_libraries(visualizeTensor TH GLFRAMEWORK TENSORVIS)

#Test visualization of protein+density
#add_executable(Test_ProteinVisualization VolumeVisualization/Test_ProteinVisualization.cpp)
#target_link_libraries (Test_ProteinVisualization TH GLFRAMEWORK PROTEINLOADER TENSORVIS)

#Test tensor math library
#add_executable(Test_TensorMathUtils Math/Test_TensorMathUtils.cpp)
#target_link_libraries (Test_TensorMathUtils TH TH_MATH)

#External function to visualize a tensor and a protein from lua
#ADD_LIBRARY(visualizeTensorAndProtein SHARED VolumeVisualization/visualizeTensorAndProtein.cpp)
#target_link_libraries(visualizeTensorAndProtein TH GLFRAMEWORK PROTEINLOADER TENSORVIS TH_MATH)


#ADD_LIBRARY(test_lua_bindings SHARED CppProteinLoader/testLuaBindings.cpp)
#TARGET_LINK_LIBRARIES(test_lua_bindings TH PROTEINLOADER)

#Convert protein coordinates with assigned atom types to binary
#add_executable(convert2bin CppProteinLoader/convert2Bin.cpp)
#target_link_libraries (convert2bin TH PROTEINLOADER)

#CUDA example
#cuda_add_executable(cuda_main CppProteinLoader/cuda_main.cu)
#target_link_libraries (cuda_main THC PROTEINLOADER)

CUDA_ADD_LIBRARY(load_protein_cuda SHARED CppProteinLoader/loadProteinCUDA.cu)
TARGET_LINK_LIBRARIES(load_protein_cuda TH THC PROTEINLOADER)

CUDA_ADD_LIBRARY(get_gradient_cuda SHARED CppProteinLoader/reprojectGradientCUDA.cu)
TARGET_LINK_LIBRARIES(get_gradient_cuda TH THC PROTEINLOADER)

CUDA_ADD_LIBRARY(load_protein_cuda_direct SHARED CppProteinLoader/loadProteinCUDAdirect.cu)
TARGET_LINK_LIBRARIES(load_protein_cuda_direct TH THC PROTEINLOADER)

#cuda_add_executable(volume_render VolumeVisualization/cudaVolumeRender.cpp VolumeVisualization/volume_render_kernel.cu)
#TARGET_LINK_LIBRARIES(volume_render TH THC GLFRAMEWORK /home/lupoglaz/NVIDIA_CUDA-7.5_Samples/common/lib/linux/x86_64/libGLEW.a)

#CUDA_ADD_EXECUTABLE(dummy_load_protein_cuda CppProteinLoader/dummyProteinLoadCuda.cu)
#TARGET_LINK_LIBRARIES(dummy_load_protein_cuda PROTEINLOADER THC)
