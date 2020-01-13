add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

add_executable(testkernel testkernel.cpp)
target_link_libraries(testkernel PRIVATE ncnn)
