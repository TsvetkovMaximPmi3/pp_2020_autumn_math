// Copyright 2020 Tsvetkov Maxim
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "./mult_mtrx_vector_vertical.h"

TEST(Parallel_Operations_MPI, Test_1) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::vector<double>> test;
    std::vector<double> test2;
    int m = 2500;  // row
    int n = 1055;  // column
    if (rank == 0) {
        test = getRandomMatrix(n, m);
        test2 = getRandomVector(m);
    }
    double time1 = MPI_Wtime();
    std::vector<double>res_p = getParallelOperations(test, n, m, test2);
    double time2 = MPI_Wtime();
    double time_res1 = time2 - time1;
    if (rank == 0) {
        double time3 = MPI_Wtime();
        std::vector<double> test3 = matrix_to_vector(test, n, m);
        std::vector<double> res_s = getSeqOperations(test3, test2, n, m, 0, n);
        double time4 = MPI_Wtime();
        double time_res2 = time4 - time3;
        std::cout << "Parallel time :" << time_res2 << std::endl;
        std::cout << "Seq time :" << time_res1 << std::endl;
        ASSERT_EQ(res_p, res_s);
    }
}
TEST(Parallel_Operations_MPI, Test_2) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::vector<double>> test;
    std::vector<double> test2;
    int m = 250;  // row
    int n = 105;  // column
    if (rank == 0) {
        test = getRandomMatrix(n, m);
        test2 = getRandomVector(m);
    }
    double time1 = MPI_Wtime();
    std::vector<double>res_p = getParallelOperations(test, n, m, test2);
    double time2 = MPI_Wtime();
    double time_res1 = time2 - time1;
    if (rank == 0) {
        double time3 = MPI_Wtime();
        std::vector<double> test3 = matrix_to_vector(test, n, m);
        std::vector<double> res_s = getSeqOperations(test3, test2, n, m, 0, n);
        double time4 = MPI_Wtime();
        double time_res2 = time4 - time3;
        std::cout << "Parallel time :" << time_res2 << std::endl;
        std::cout << "Seq time :" << time_res1 << std::end;
        ASSERT_EQ(res_p, res_s);
    }
}
TEST(Parallel_Operations_MPI, Test_3) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::vector<double>> test;
    std::vector<double> test2;
    int m = 25;  // row
    int n = 105;  // column
    if (rank == 0) {
        test = getRandomMatrix(n, m);
        test2 = getRandomVector(m);
    }
    double time1 = MPI_Wtime();
    std::vector<double>res_p = getParallelOperations(test, n, m, test2);
    double time2 = MPI_Wtime();
    double time_res1 = time2 - time1;
    if (rank == 0) {
        double time3 = MPI_Wtime();
        std::vector<double> test3 = matrix_to_vector(test, n, m);
        std::vector<double> res_s = getSeqOperations(test3, test2, n, m, 0, n);
        double time4 = MPI_Wtime();
        double time_res2 = time4 - time3;
        std::cout << "Parallel time :" << time_res2 << std::endl;
        std::cout << "Seq time :" << time_res1 << std::endl;
        ASSERT_EQ(res_p, res_s);
    }
}
TEST(Parallel_Operations_MPI, Test_4) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::vector<double>> test;
    std::vector<double> test2;
    int m = 200;  // row
    int n = 155;  // column
    if (rank == 0) {
        test = getRandomMatrix(n, m);
        test2 = getRandomVector(m);
    }
    double time1 = MPI_Wtime();
    std::vector<double>res_p = getParallelOperations(test, n, m, test2);
    double time2 = MPI_Wtime();
    double time_res1 = time2 - time1;
    if (rank == 0) {
        double time3 = MPI_Wtime();
        std::vector<double> test3 = matrix_to_vector(test, n, m);
        std::vector<double> res_s = getSeqOperations(test3, test2, n, m, 0, n);
        double time4 = MPI_Wtime();
        double time_res2 = time4 - time3;
        std::cout << "Parallel time :" << time_res2 << std::endl;
        std::cout << "Seq time :" << time_res1 << std::endl;
        ASSERT_EQ(res_p, res_s);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
