#include "defs.h"

#include <stdint.h>

#include <thrust/device_vector.h>

__device__ __forceinline__
uint64_t random(uint64_t seed) {
    return (seed * 0x5DEECE66DL + 0xBL) & ((1LL << 48) - 1);
}

__device__
int get_global_id()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
bool get_bit(uint32_t var, const bit_mask_t *bit_mask)
{
    uint32_t offset = var / 32;
    uint32_t bit_offset = var % 32;
    size_t word_offset = offset + bit_mask->n_words * get_global_id();

    uint32_t word = bit_mask->data[word_offset];

    return (word << (31 - bit_offset)) >> 31;
}

__device__
void flip_bit(uint32_t var, const bit_mask_t *bit_mask)
{
    uint32_t offset = var / 32;
    uint32_t bit_offset = var % 32;
    size_t word_offset = offset + bit_mask->n_words * get_global_id();

    uint32_t word = bit_mask->data[word_offset];
    word ^= 1 << bit_offset;

    bit_mask->data[word_offset] = word;
}

__device__
bool evaluate_literal(
    uint32_t lit,
    const bit_mask_t *bit_mask,
    uint32_t flip_var)
{
    bool sign = lit & 1;
    uint32_t var = lit >> 1;

    uint32_t offset = var / 32;
    uint32_t bit_offset = var % 32;

    uint32_t word = bit_mask->data[offset + bit_mask->n_words * get_global_id()];
    uint32_t evaluation = (word << (31 - bit_offset)) >> 31;

    bool flip = var == flip_var;
    return (evaluation ^ sign) ^ flip;
}

__device__
bool evaluate_clause(
    const uint32_t *clauses,
    uint32_t clause_index,
    const bit_mask_t *bit_mask,
    uint32_t flip_var)
{
    // this function behaved weird and therefore this strange return pattern
    // there was a bug in one of the calling functions, might have been due to that

    uint32_t offset = clauses[clause_index];
    uint32_t n_lits = clauses[offset];

    for (int i = 0; i < n_lits; i++){
        uint32_t literal = clauses[offset + 1 + i];

        if (evaluate_literal(literal, bit_mask, flip_var)) {
            return true;
        }
    }

    return false;
}

__device__
uint32_t count_unsat(
    uint32_t var,
    const shared_context_t *ctx,
    bool flip)
{
    uint32_t offset = ctx->indices[var];
    uint32_t n_clauses = ctx->indices[offset];

    uint32_t evaluation = 0;
    for (size_t i = 0; i < n_clauses; i++){
        uint32_t index = ctx->indices[offset + 1 + i];

        evaluation += !evaluate_clause(
                                ctx->clauses,
                                index,
                                &ctx->bit_mask,
                                (flip) ? var : 0);
    }

    return evaluation;
}

__device__
int find_clause(uint32_t index, uint32_t *unsat, uint32_t n_unsat) {
    for (size_t i = 0; i < n_unsat; i++) {
        if (unsat[i] == index) {
            return i;
        }
    }

    return -1;
}

__device__
void add_unsat_clause(uint32_t clause_index, uint32_t *unsat, uint32_t n_unsat) {
    // here still must be checked if new size does not violate buffer size
    unsat[n_unsat] = clause_index;
}

__device__
void remove_unsat_clause(uint32_t clause_index, uint32_t *unsat, uint32_t n_unsat) {
    int index = find_clause(clause_index, unsat, n_unsat);
    if (index > 0) {
        unsat[index] = unsat[n_unsat - 1];
    }
}

__device__
uint32_t count_unsat_update_cache(
    uint32_t var,
    const shared_context_t *ctx,
    uint32_t *unsat,
    uint32_t n_unsat)
{
    uint32_t offset = ctx->indices[var];
    uint32_t n_clauses = ctx->indices[offset];

    uint32_t unsat_counter = n_unsat;
    for (size_t i = 0; i < n_clauses; i++) {
        uint32_t index = ctx->indices[offset + 1 + i];

        bool flipped = evaluate_clause(ctx->clauses, index, &ctx->bit_mask, var);
        bool unflipped = evaluate_clause(ctx->clauses, index, &ctx->bit_mask, 0);

        if (!flipped && unflipped) {
            // this is a new unsat clause => add it
            add_unsat_clause(index, unsat, unsat_counter);
            unsat_counter += 1;
        }
        else if (flipped && !unflipped) {
            // this is a new sat clause => remove it
            remove_unsat_clause(index, unsat, unsat_counter);
            unsat_counter -= 1;
        }
        else {
            // both cases are sat => nothing must be added
            // both cases are unsat, already in list
            continue;
        }
    }

    return unsat_counter;
}

__device__
bool evaluate_clause_weight(
    const uint32_t* clauses,
    uint32_t clause_index,
    const bit_mask_t *bit_mask,
    uint32_t flip_var,
    uint32_t t)
{
    // is equivalent to counting clauses going from unsat to sat if t == 1

    uint32_t offset = clauses[clause_index];
    uint32_t n_lits = clauses[offset];

    bool state = false;
    uint32_t c = 0;

    for (int i = 0; i < n_lits; i++) {
        uint32_t lit = clauses[offset + 1 + i];
        bool eval = evaluate_literal(lit, bit_mask, 0);

        // if variable to flip is current variable
        // then save state of variable to flip
        if (flip_var == (lit >> 1)) {
            state = eval;
        }

        // count all true evaluations
        if (eval) {
            c++;
        }
    }

    // if variable to flip evaluates to false, then it does not contribute to c
    // therefore, flipping would increase number of sat literals
    // else flipping would decrease number of sat literals

    return c + ((state) ? -1 : 1) == t;
}

__device__
uint32_t evaluate_weight(
    uint32_t var,
    const shared_context_t *ctx,
    uint32_t t)
{
    uint32_t offset = ctx->indices[var];
    uint32_t n_clauses = ctx->indices[offset];

    uint32_t evaluation = 0;
    for (size_t i = 0; i < n_clauses; i++) {
        uint32_t index = ctx->indices[offset + 1 + i];
        evaluation += evaluate_clause_weight(
            ctx->clauses, index, &ctx->bit_mask, var, t);
    }

    return evaluation;
}

#define UNSAT_ARRAY_SIZE 512
#define UNSAT_ARRAY_MAGIC_SIZE (UNSAT_ARRAY_SIZE - 64)

__global__
void local_search_k(
    thread_context_t *threads,
    const shared_context_t ctx,
    const size_t n_clauses,
    const size_t n_variables,
    const size_t iterations,
    const double flip_probability)
{
    thread_context_t *thread_ctx = threads + get_global_id();

    __shared__ uint32_t unsat_cache[UNSAT_ARRAY_SIZE];

    size_t n_sat = 0;
    for (size_t i = 0; i < n_clauses; i++) {
        n_sat += evaluate_clause(ctx.clauses, i, &ctx.bit_mask, 0);
    }

    size_t n_unsat = n_clauses - n_sat;
    size_t n_unsat_before_main_loop = n_unsat;

    // some space left if number of unsat clauses increases
    if (n_unsat <= UNSAT_ARRAY_MAGIC_SIZE){
        // fill array with indices of unsat clauses
        size_t counter = 0;
        for (size_t i = 0; i < n_clauses; i++){
            if (!evaluate_clause(ctx.clauses, i, &ctx.bit_mask, 0)) {
                unsat_cache[counter] = i;
                counter++;
            }
        }
    }

    for (size_t i = 0; i < iterations; i++) {
        if (!n_unsat) {
            break;
        }

        uint32_t clause_index;
        if (n_unsat > UNSAT_ARRAY_MAGIC_SIZE) {
            // get random clause
            thread_ctx->random = random(thread_ctx->random);
            clause_index = thread_ctx->random % n_clauses;
        }

        else {
            // get random clause from list
            thread_ctx->random = random(thread_ctx->random);
            clause_index = unsat_cache[thread_ctx->random % n_unsat];
        }

        if (evaluate_clause(ctx.clauses, clause_index, &ctx.bit_mask, 0)) {
            continue;
        }

        size_t offset = ctx.clauses[clause_index];
        uint32_t n_lits = ctx.clauses[offset];

        // greedy or random
        thread_ctx->random = random(thread_ctx->random);
        uint32_t flip = thread_ctx->random % UINT32_MAX;

        if (flip <= flip_probability * UINT32_MAX) {
            // flip random bit
            thread_ctx->random = random(thread_ctx->random);
            uint32_t random_bit = thread_ctx->random % n_lits;

            uint32_t literal = ctx.clauses[offset + 1 + random_bit];
            uint32_t var = literal >> 1;

            int before = count_unsat(var, &ctx, false);
            int after  = count_unsat(var, &ctx, true);

            // if there are only a few unsat clauses left, update list
            if (n_unsat <= UNSAT_ARRAY_MAGIC_SIZE) {
                count_unsat_update_cache(
                    var, &ctx, unsat_cache, n_unsat);
            }

            flip_bit(var, &ctx.bit_mask);

            n_unsat += after - before;
        }
        else {
            // find literal, flipping of which produces
            // the minimal number of unsat clauses

            int n_unsat_before = -1;
            int n_unsat_after = -1;
            uint32_t var_after = 0;
            int lmk_score = -1;

            for (size_t j = 0; j < n_lits; j++) {
                uint32_t literal = ctx.clauses[offset + 1 + j];
                uint32_t var = literal >> 1;

                int before = count_unsat(var, &ctx, false);
                int after  = count_unsat(var, &ctx, true);

                if (after > before) {
                    continue;
                }

                if (n_lits > 3) {
                    // if there are ties, then break them with lweight
                    if (n_unsat_after == -1 || after == n_unsat_after) {
                        int weight1 = evaluate_weight(var, &ctx, 1);
                        int weight2 = evaluate_weight(var, &ctx, 2);
                        int score = (n_lits - 2) * weight1 + (n_lits - 3) * weight2;
                        if (lmk_score == -1 || score > lmk_score) {
                            lmk_score = score;
                            var_after = var;
                            n_unsat_before = before;
                            n_unsat_after = after;
                        }
                    }
                }

                // this we do always if we find literal with strictly less unsat clauses
                if (n_unsat_after == -1 || after < n_unsat_after) {
                    lmk_score = -1;
                    var_after = var;
                    n_unsat_before = before;
                    n_unsat_after = after;
                }
            }

            if (n_unsat_after != -1) {
                // if there are only a few unsat clauses left, update list
                if (n_unsat <= UNSAT_ARRAY_MAGIC_SIZE) {
                    count_unsat_update_cache(
                        var_after, &ctx, unsat_cache, n_unsat);
                }

                n_unsat = n_unsat - n_unsat_before + n_unsat_after;
                flip_bit(var_after, &ctx.bit_mask);
            }
        }

        if (n_unsat < n_unsat_before_main_loop) {
            break;
        }
    }

    thread_ctx->n_sat = n_clauses - n_unsat;

}

void local_search(
    thread_context_t *threads,
    const shared_context_t *ctx,
    const size_t n_clauses,
    const size_t n_variables,
    const size_t iterations,
    const double flip_probability,
    const size_t n_threads)
{
    size_t threads_per_block = 32;
    size_t blocks = n_threads / threads_per_block;

    local_search_k<<<blocks, threads_per_block>>>(
        threads,
        *ctx,
        n_clauses,
        n_variables,
        iterations,
        flip_probability
    );
}
