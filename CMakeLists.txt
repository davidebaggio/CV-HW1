cmake_minimum_required(VERSION 3.16)
PROJECT (opencv_test)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED )
set(LIB_SRC
	src/dbscan.cpp
	src/orb_detector.cpp
	src/sift_detector.cpp
)

set(HEADERS
	include/dbscan.hpp
	include/orb_detector.hpp
	include/sift_detector.hpp
)

add_library(image_lib STATIC ${LIB_SRC} ${HEADERS})
target_link_libraries(image_lib ${OpenCV_LIBS})

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin )
#add_executable( generate_negative src/generate_negative.cpp )
#add_executable( generate_annotations src/generate_annotations.cpp )
#add_executable( detect_object src/detect_object.cpp )
add_executable( test_images_detection src/test_images_detection.cpp )
#add_executable( dbscan src/dbscan.cpp )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
#target_link_libraries(generate_negative ${OpenCV_LIBS})
#target_link_libraries(generate_annotations ${OpenCV_LIBS})
#target_link_libraries(detect_object ${OpenCV_LIBS})
target_link_libraries(test_images_detection image_lib ${OpenCV_LIBS})
#target_link_libraries(dbscan ${OpenCV_LIBS})