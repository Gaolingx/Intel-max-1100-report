// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <vector>
#if defined(MANAGED)
#include <thrust/universal_vector.h>
#else
#include <thrust/device_vector.h>
#endif

#include "Stream.h"

#define IMPLEMENTATION_STRING "Thrust"

template <class T>
class ThrustStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

  #if defined(MANAGED)
    thrust::universal_vector<T> a, b, c;
  #else
    thrust::device_vector<T> a, b, c;
    std::vector<T> h_a, h_b, h_c;
  #endif

  public:
    ThrustStream(BenchId bs, const intptr_t array_size, const int device_id,
		 T initA, T initB, T initC);
    ~ThrustStream() = default;

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC);
};

