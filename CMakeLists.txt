project( ceres_curve_fitting )

set( CMAKE_CXX_FLAGS "-std=c++11 -w -g" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
    

find_package( OpenCV REQUIRED )
find_package(G2O REQUIRED)

include_directories( ${OpenCV_DIRS} )
include_directories(${G2O_INCLUDE_DIR} "/usr/local/include/eigen3/")


add_executable(vo vo.cpp)

target_link_libraries( vo ${OpenCV_LIBS} ${G2O_LIBS} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)    

