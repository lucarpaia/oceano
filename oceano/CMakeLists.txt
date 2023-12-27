# oceano/CMakeLists.txt

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/lib/include
  )

ADD_EXECUTABLE(oceano source/main.cc)
DEAL_II_SETUP_TARGET(oceano)

#TARGET_LINK_LIBRARIES(oceano lib)
TARGET_LINK_LIBRARIES(oceano)
