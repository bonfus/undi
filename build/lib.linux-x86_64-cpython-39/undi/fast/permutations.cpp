/* ===  Compute indexes and permutaions === */

#include <cstddef>
#include <omp.h>
#include "permutations.hpp"

/* computes the dimensions of the Hilbert space */
size_t hilbert_space_dims(size_t nspins, size_t * dims_ptr) {
    size_t hdim = 1;
    for (size_t i = 0; i < nspins; i++)
        hdim *= dims_ptr[i];
    return hdim;
}

/* translate a multidimensional index into a flat one */
void flat_to_nd_idx(size_t n, size_t* start, size_t * pdims, size_t * prods, size_t nb ){
    size_t c;

    for(size_t i = nb; i > 0;) {
        --i;

        c = n % pdims[i];
        n = n / pdims[i];

        start[i] = c;
    }
}


void print (size_t * res, size_t * prods, size_t nd) {
/* FOR DEBUG
    size_t l = 0;
    for (size_t i = 0; i < nd; i++) {
        l += res[i] * prods[i];
    }
    cout << l << endl;
*/
}


void swap(size_t *arr, size_t a, size_t b) {
    size_t tmp = arr[b];
    arr[b] = arr[a];
    arr[a] = tmp;
}

void omp_split_computation(size_t hdim, size_t * dims_ptr, size_t * prods_ptr, size_t nspins, size_t * idx_nd, size_t &start, size_t &end) {

    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    if (nthreads == 1) {
        // serial version is easy, start from 0
        for (size_t i=0; i < nspins; i++)
            idx_nd[i] = 0;
        start = 0;
        end = hdim;
        return;
    }

    size_t chunk = hdim / nthreads;
    start = chunk * tid;
    end = start + chunk;

    if ( tid == nthreads - 1)
        end += hdim % nthreads;

    // compute starting value
    flat_to_nd_idx(start, idx_nd, dims_ptr, prods_ptr, nspins);
}

/* This function computes the index of elements in a permutation of the
 * tensor elements bringing spin "spin" to the second last place.
 *
 * The two index correspond to
 *
 *     b[i] = a[idx[i]]
 *
 * and
 *
 *     b[invidx[i]] = a[i]
 *
 */
void compute_idx_invidx(size_t hdim, size_t * dims_ptr, size_t nspins, size_t spin, size_t * idx_ptr, size_t * inv_idx_ptr) {
    // compute index - init part
    size_t prods_ptr[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;

    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);
#pragma omp parallel default(none) shared(idx_ptr, inv_idx_ptr, dims_ptr, prods_ptr, nspins, hdim)
{
    size_t idx_nd[nspins], start, end;
    omp_split_computation(hdim, dims_ptr, prods_ptr, nspins, idx_nd, start, end);

    size_t l;
    for (size_t i=start; i < end; i++) {
        l = fill(idx_nd, prods_ptr, nspins);
        idx_ptr[i] = l;
        inv_idx_ptr[l] = i;
        iterate2(nspins, dims_ptr, idx_nd);
    }
} // end omp parallel
    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);

}


void compute_idx(size_t hdim, size_t * dims_ptr, size_t nspins, size_t spin, size_t * idx_ptr) {
    // compute index - init part
    size_t prods_ptr[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;

    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);

#pragma omp parallel default(none) shared(idx_ptr, dims_ptr, prods_ptr, nspins, hdim)
{
    size_t idx_nd[nspins], start, end;
    omp_split_computation(hdim, dims_ptr, prods_ptr, nspins, idx_nd, start, end);

    // loop over dimension
    for (size_t i=start; i < end; i++) {
        idx_ptr[i] = fill(idx_nd, prods_ptr, nspins);
        iterate2(nspins, dims_ptr, idx_nd);
    }
} // end omp parallel

    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);

}


void compute_invidx(size_t hdim, size_t * dims_ptr, size_t nspins, size_t spin, size_t * inv_idx_ptr) {
    // compute index - init part
    size_t prods_ptr[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;

    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);

#pragma omp parallel default(none) shared(inv_idx_ptr, dims_ptr, prods_ptr, nspins, hdim)
{
    size_t idx_nd[nspins], start, end;
    omp_split_computation(hdim, dims_ptr, prods_ptr, nspins, idx_nd, start, end);


    size_t l;
    for (size_t i=start; i < end; i++) {
        l = fill(idx_nd, prods_ptr, nspins);
        inv_idx_ptr[l] = i;
        iterate2(nspins, dims_ptr, idx_nd);
    }
} // end omp
    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);

}

/* End compute indexes and permutaions */
