/* ===  Compute indexes and permutaions === */

size_t hilbert_space_dims(size_t nspins, size_t * dims_ptr);

void flat_to_nd_idx(size_t n, size_t* start, size_t * pdims, size_t * prods, size_t nb );

void print (size_t * res, size_t * prods, size_t nd);

void compute_idx_invidx(size_t hdim, size_t * dims_ptr, size_t nspins, size_t spin, size_t * idx_ptr, size_t * inv_idx_ptr);

void compute_idx(size_t hdim, size_t * dims_ptr, size_t nspins, size_t spin, size_t * idx_ptr);

void compute_invidx(size_t hdim, size_t * dims_ptr, size_t nspins, size_t spin, size_t * inv_idx_ptr);

void swap(size_t *arr, size_t a, size_t b);

/* Inlined functions */

inline size_t fill (size_t * res, size_t * prods, size_t nd) {
    size_t l = 0;
    for (size_t i = 0; i < nd; i++) {
        l += res[i] * prods[i];
    }
    return l;
}


inline void iterate(size_t d, size_t nd, size_t * dims, size_t * start, size_t * res, size_t * prods, size_t *idx, size_t * arr, size_t na) {
    if (d >= nd) { //stop clause
       //print(res,prods,nd);
       arr[*idx] = fill(res,prods,nd);
       (*idx)++;
       return;
   }
   for (size_t i = start[d]; i < dims[d]; i++) {
       res[d] = i;
       start[d] = 0;
       iterate(d+1, nd, dims, start, res, prods, idx, arr, na);
       if (*idx == na)
           break;
   }
}


inline void iterate2(size_t dimensions, size_t* maximums, size_t* ordinates)
{
    // iterate over dimensions in reverse...
    for (int dimension = dimensions - 1; dimension >= 0; dimension--)
    {

        if (ordinates[dimension] < maximums[dimension]-1)
        {
            // If this dimension can handle another increment... then done.
            ordinates[dimension]++;
            break;
        }

        // Otherwise, reset this dimension and bubble up to the next dimension to take a look
        ordinates[dimension] = 0;
    }
}
