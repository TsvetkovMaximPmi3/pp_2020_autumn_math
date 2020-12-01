// Copyright 2020 Tsvetkov Maxim
#pragma once
#ifndef MODULES_TASK_2_TSVETKOV_M_MULT_MTRX_VECTOR_VERTICAL_MULT_MTRX_VECTOR_VERTICAL_H_
#define MODULES_TASK_2_TSVETKOV_M_MULT_MTRX_VECTOR_VERTICAL_MULT_MTRX_VECTOR_VERTICAL_H_
#include <vector>
#include <string>

std::vector<std::vector<double>> getRandomMatrix(int m, int n);



void transpose(std::vector<std::vector<double>>* a);

std::vector<double> matrix_to_vector(std::vector<std::vector<double>> a,
    int m, int n);



std::vector<double> getSeqOperations(std::vector<double> a, std::vector<double>b,
    int n, int m, int start_pos, int finish_pos);

std::vector<double> getParallelOperations(std::vector<std::vector<double>> a,
    int n, int m, std::vector<double>b);
std::vector<double> getRandomVector(int n);
#endif  // MODULES_TASK_2_TSVETKOV_M_MULT_MTRX_VECTOR_VERTICAL_MULT_MTRX_VECTOR_VERTICAL_H_
