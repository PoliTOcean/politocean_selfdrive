cmake_minimum_required( VERSION 2.8 )
project( main )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )
set(OpenCV_DIR /usr/local/opencv/share/OpenCV)
# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( main main.cpp )
# 链接OpenCV库
target_link_libraries( main ${OpenCV_LIBS} )