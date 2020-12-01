// Copyright 2020 Tsvetkov Maxim
#include <mpi.h>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include "../../../modules/task_2/tsvetkov_m_mult_mtrx_vector_vertical/mult_mtrx_vector_vertical.h"

std::vector<std::vector<double>> getRandomMatrix(int n, int m) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<std::vector<double>> a;
    a.assign(n, std::vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            a[i][j] = (gen() % 9) + 1;
    }
    return a;
}
std::vector<double> getRandomVector(int n) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> a(n);
    for (int j = 0; j < n; j++) {
        a[j] = (gen() % 9) + 1;
    }
    return a;
}




std::vector<double> matrix_to_vector(std::vector<std::vector<double>> a,
    int n, int m) {
    int size = m * n;
    std::vector<double> res(size);
    int tmp = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res[tmp] = a[i][j];
            tmp++;
        }
    }
    return res;
}

std::vector<double> getSeqOperations(std::vector<double> a, std::vector<double>b, int n,
    int m, int start_pos, int finish_pos) {
    std::vector<double> globalvec(n);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank >= n) {
        for (int i = 0; i < n; i++) {
            globalvec[i] = 0;
        }
        return globalvec;
    }
    int pos = start_pos;
    int index = finish_pos - start_pos;
    for (size_t i = 0; i < index; i++) {
        for (size_t j = 0; j < m; j++) {
            globalvec[pos] += a[j + i * m] * b[j];
        }
        pos++;
    }
    return globalvec;
}


void transpose(std::vector<std::vector<double>>* a) {
    int m = a->size();
    int n = (*a)[0].size();
    std::vector<std::vector<double>> res;
    res.assign(n, std::vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            res[j][i] = (*a)[i][j];
    }
    *a = res;
}

std::vector<double> getParallelOperations(std::vector<std::vector<double>> a,
    int n, int m, std::vector<double>b) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //  transpose(&a);
    std::vector<double> res;
    if (rank == 0)res = matrix_to_vector(a, n, m);
    int delta = n;
    int k = 0;
    if (rank != 0)b = std::vector<double>(m);
    int part_to_proc = delta / size;
    int part_to_last_proc = delta % size;
    MPI_Bcast(&b[0], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> local_res;
    if (n >= size) {
        if (rank == 0) {
            for (int proc = 1; proc < size; proc++) {
                if (proc < size - 1) {
                    MPI_Send(&res[0] + proc * part_to_proc * m, part_to_proc * m,
                        MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                } else {
                    if (part_to_last_proc != 0) {
                        MPI_Send(&res[0] + res.size() - ((part_to_last_proc + part_to_proc) * m),
                            (part_to_last_proc + part_to_proc) * m,
                            MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                    } else {
                        MPI_Send(&res[0] + proc * part_to_proc * m, part_to_proc * m,
                            MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }

        std::vector<double> local_vec(part_to_proc * m);
        if (rank == size - 1)local_vec.resize(part_to_proc * m + part_to_last_proc * m);
        if (rank == 0) {
            local_vec = std::vector<double>(res.begin(),
                res.begin() + part_to_proc * m);
        } else {
            MPI_Status status;
            if (rank < size - 1) {
                MPI_Recv(&local_vec[0], part_to_proc * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            } else {
                if (part_to_last_proc != 0) {
                    MPI_Recv(&local_vec[0], (part_to_last_proc + part_to_proc) * m, MPI_DOUBLE,
                        0, 0, MPI_COMM_WORLD, &status);
                } else {
                    MPI_Recv(&local_vec[0], part_to_proc * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
                }
            }
        }
        if (rank < size - 1)local_res = getSeqOperations(local_vec, b, n, m,
            part_to_proc * rank, part_to_proc + (part_to_proc * rank));
        else
            local_res = getSeqOperations(local_vec, b, n, m,
                part_to_proc * rank, delta);
    } else {
        if (rank == 0) {
            for (int proc = 1; proc < n; proc++) {
                MPI_Send(&res[0] + proc * m, m,
                    MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            }
        }
        std::vector<double> local_vec(m);
        if (rank == 0) {
            local_vec = std::vector<double>(res.begin(),
                res.begin() + m);
        } else {
            if (rank < n) {
                MPI_Status status;
                MPI_Recv(&local_vec[0], m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            }
        }
        local_res = getSeqOperations(local_vec, b, n, m,
            rank, (rank + 1));
    }
    std::vector<double> global_vec(n);
    MPI_Reduce(local_res.data(), global_vec.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_vec;
}
