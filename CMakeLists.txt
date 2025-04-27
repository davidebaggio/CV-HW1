cmake_minimum_required(VERSION 3.16)
PROJECT (opencv_test)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED )
set(LIB_SRC
	src/dbscan.cpp
	src/haar_detector.cpp
	src/orb_detector.cpp
	src/sift_detector.cpp
	src/detection.cpp
)

set(HEADERS
	include/dbscan.hpp
	include/haar_detector.hpp
	include/orb_detector.hpp
	include/sift_detector.hpp
	include/detection.hpp
)

add_library(image_lib STATIC ${LIB_SRC} ${HEADERS})
target_link_libraries(image_lib ${OpenCV_LIBS})

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin )
add_executable( test_images_detection src/test_images_detection.cpp )
add_executable( performance src/performance.cpp )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
target_link_libraries(test_images_detection image_lib ${OpenCV_LIBS})
target_link_libraries(performance image_lib ${OpenCV_LIBS})