#include <fstream>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using literal_t = int;
using variable_t = size_t;
using variable_map = std::map<variable_t, variable_t>;
using clause_t = std::vector<literal_t>;
using clause_vector = std::vector<clause_t>;

struct program_options_t {
    std::string input;
    uint32_t n_threads;
    uint32_t n_iterations;
    double flip_probability;
};

struct thread_context_t {
    uint64_t random;
    uint32_t n_sat;
};

struct bit_mask_t {
    uint32_t *data;
    size_t n_words;
};

struct shared_context_t {
    uint32_t *indices;
    uint32_t *clauses;
    bit_mask_t bit_mask;
};

program_options_t get_program_options(int argc, char *argv[]);

std::tuple<variable_map, clause_vector>
    read_cnf(std::string const& filename);

void local_search(
    thread_context_t *threads,
    const shared_context_t *ctx,
    const size_t n_clauses,
    const size_t n_variables,
    const size_t iterations,
    const double flip_probability,
    const size_t n_threads);
