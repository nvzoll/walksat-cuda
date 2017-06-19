#include "defs.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "stopwatch.cuh"

thrust::host_vector<uint32_t>
serialize_clauses(clause_vector const& clauses)
{
    size_t size = 0;
    for (clause_t const& clause : clauses) {
        size += clause.size();
    }

    size += clauses.size() * 2;

    // create the clause buffer which consists of n_clauses pointers and space
    // for non-trivial and trivial clauses
    thrust::host_vector<uint32_t> info(size);

    size_t offset = 0;
    size_t i = 0;
    for (clause_t const& clause : clauses) {
        // save offset of clause
        info[i] = clauses.size() + offset;

        // save size of clause
        info[clauses.size() + offset] = clause.size();

        offset++;

        for (literal_t l : clause) {
            uint32_t var = abs(l);
            uint32_t sign = (l < 0);

            // test if index uses at most 31 bits
            // because we need the first bit for the sign
            assert(var < ((1U << 31) - 1));

            // save literal at corresponding position
            info[clauses.size() + offset] = (var << 1) | sign;
            offset++;
        }

        i++;
    }

    return info;
}

thrust::host_vector<uint32_t>
serialize_clause_indices(clause_vector& clauses, size_t n_vars)
{
    thrust::host_vector<thrust::host_vector<size_t>> vars(n_vars);

    // For every var, save clauses in which it appears
    size_t i = 0;
    for (clause_t const& clause : clauses) {
        for (literal_t literal : clause) {
            size_t var = abs(literal);
            vars[var].push_back(i);
        }
        i++;
    }

    size_t size = 0;
    for (auto const& v : vars) {
        size += v.size();
    }

    size += n_vars * 2;

    thrust::host_vector<uint32_t> info(size);

    size_t offset = 0;
    i = 0;
    for (auto const& vec : vars) {
        // save index where list of clause indices is stored
        info[i] = n_vars + offset;
        // save number of clauses
        info[n_vars + offset] = vec.size();
        offset++;

        // save list of clause indices
        for (size_t indice : vec) {
            info[n_vars + offset] = indice;
            offset++;
        }

        i++;
    }

    return info;
}

int main(int argc, char *argv[])
{
    program_options_t options = get_program_options(argc, argv);

    variable_map variables;
    clause_vector clauses;

    std::tie(variables, clauses) = read_cnf(options.input);

    printf("CNF read: %zu clauses, %zu variables\n", clauses.size(), variables.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, ~0);

    size_t n_vars = variables.size() + 1;
    size_t n_bit_words = (n_vars / 32) + 1;

    thrust::host_vector<uint32_t> host_bit_mask(n_bit_words * options.n_threads);

    for (size_t i = 0; i < options.n_threads; ++i) {
        // variable 0 is always false, i.e., the most right bit is zero
        host_bit_mask[i * n_bit_words + 0] = (static_cast<uint32_t>(dist(gen)) >> 1) << 1;

        // the rest are random
        for (size_t j = 1; j < n_bit_words; ++j) {
            host_bit_mask[i * n_bit_words + j] = static_cast<uint32_t>(dist(gen));
        }
    }

    thrust::host_vector<thread_context_t> host_threads(options.n_threads);
    for (auto& thread : host_threads) {
        thread.random = dist(gen);
    }

    thrust::device_vector<thread_context_t> dev_threads = host_threads;
    thrust::device_vector<uint32_t> dev_bit_mask = host_bit_mask;
    thrust::device_vector<uint32_t> dev_clauses = serialize_clauses(clauses);
    thrust::device_vector<uint32_t> dev_clause_indices = serialize_clause_indices(clauses, n_vars);

    shared_context_t search_ctx = {
        thrust::raw_pointer_cast(&dev_clause_indices[0]),
        thrust::raw_pointer_cast(&dev_clauses[0]),
        {
            thrust::raw_pointer_cast(dev_bit_mask.data()),
            n_bit_words
        }
    };

    size_t passes = 0;
    float elapsedTime = 0.;
    CudaStopwatch stopwatch;

    while (true) {
        stopwatch.start();
        local_search(
            thrust::raw_pointer_cast(dev_threads.data()),
            &search_ctx,
            clauses.size(),
            n_vars,
            options.n_iterations,
            options.flip_probability,
            options.n_threads
        );
        elapsedTime += stopwatch.stop();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed with the following "
                "message\n %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        thrust::host_vector<thread_context_t> threads = dev_threads;
        std::vector<uint32_t> n_sat(threads.size());

        bool found = std::any_of(std::begin(threads), std::end(threads),
            [&clauses, &n_sat] (thread_context_t const& ctx) -> bool
        {
            n_sat.push_back(ctx.n_sat);
            return ctx.n_sat == clauses.size();
        });

        if (found || passes >= 1000) {
            break;
        }

        uint32_t best = *std::max_element(std::begin(n_sat), std::end(n_sat));
        printf("%4zu levels of irony, %u/%zu\n", passes, best, clauses.size());

        passes++;
    }

    printf("Ended up on %zu levels my dude\n", passes);
    printf("Total time on GPU: %f ms\n", elapsedTime);

    return 0;
}
