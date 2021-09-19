add_executable(Demo.X Demo.cpp)
target_link_libraries(Demo.X PRIVATE STKFMM_STATIC ${DEP_LIB}
                                     MPI::MPI_CXX)

add_executable(Demo_SK.X Demo_SK.cpp ../Test/SimpleKernel.cpp)
target_link_libraries(Demo.X PRIVATE ${DEP_LIB} MPI::MPI_CXX)
