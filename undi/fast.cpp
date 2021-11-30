#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <iostream>


#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

namespace py = pybind11;
using namespace std;

/*
 * Do not try to understand this code. Check the python implementation instead.
 * It's in celio_on_steroids.
 *
 */

size_t permute(size_t idx, size_t * dims, size_t * prod, size_t * perm, size_t ndims) {
    // This is the same as np.swapaxes.
    // idx is the required index,
    // dims are the dimensions of the various axes
    // prod is:
    //           for (size_t i=0;i<ndims;i++){
    //               prod[i]=1;
    //               for (size_t j=i+1;j<ndims;j++)
    //                   prod[i] *= dims[j];
    //           }
    // perm is the permutation required
    // ndims is the total number of dimensions
    size_t r;
    size_t total = prod[0]*dims[0];
    r = 0;
    for (size_t i=0; i<ndims; i++) {
        total /= dims[perm[i]];
        r += (idx / total) * prod[perm[i]];
        idx = idx % total;
    }
    return r;
}
size_t permute2(size_t idx, size_t * dims, size_t * prod, size_t total, size_t ndims) {
    // This is the same as np.swapaxes.
    // idx is the required index,
    // dims are the dimensions of the various axes already permuted,
    // prod is obtained from:
    //           for (size_t i=0;i<ndims;i++){
    //               prod[i]=1;
    //               for (size_t j=i+1;j<ndims;j++)
    //                   prod[i] *= dims[j];
    //           }
    // and later permuted (assumed to be already permuted in input)
    // ndims is the total number of dimensions
    size_t r= 0;
    for (size_t i=0; i<ndims; i++) {
        total /= dims[i];
        r += (idx / total) * prod[i];
        idx = idx % total;
    }
    return r;
}

void gen_idxes(size_t * dims, size_t * prod, size_t ndims, size_t * out) {
    /*
     * dims:  the dimension of the various axes
     * prod:  permuted product of dimensions
     * ndims: total numer of dimensions
     * out:   the list with the indexes
     */

    size_t coord[ndims];
    size_t total=1;
    for (size_t i=0; i<ndims;i++){
        coord[i] = 0;
        total *= dims[i];
    }

    for (size_t l=0; l<total; l++)
    {
        out[l] = 0;
        for (size_t d = 0; d < ndims; d++)
            out[l] += coord[d] * prod[d];

        // increment relevant index
        for(size_t j = ndims-1 ; j>=0 ; j--)
        {
            if(++coord[j]<dims[j])
                break;
            else
                coord[j]=0;
        }
    }
}

template <typename T>
double measure(py::array_t<std::complex<T>, py::array::c_style> op, py::array_t<std::complex<T>, py::array::c_style> psi) {
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong");

    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);

    size_t steps = psi_buf.shape[0]/op_buf.shape[0];
    size_t op_dim = op_buf.shape[0];

    double result = 0.0;

#pragma omp parallel reduction(+:result) default(none) shared(steps,psi_ptr,op_ptr) firstprivate(op_dim)
{
    size_t s;
    std::complex<T> *r0 = new std::complex<T>[op_dim];

    // loop on slices
#pragma omp for
    for (size_t step = 0; step < steps; step++) {
        // memset r0
        for (size_t i = 0; i < op_dim; i++)
            r0[i] = {0.0,0.0};

        // now the dot product
        s = step * op_dim;

        // dot product
        for (size_t i = 0; i < op_dim; i++) {
            for (size_t j = 0; j < op_dim; j++) {
                r0[i] += op_ptr[i*op_dim + j] * psi_ptr[s+j];
            }
        }

        for (size_t i = 0; i < op_dim; i++)
            result += ( conj(psi_ptr[s+i]) * r0[i] ).real() ;
    }
    delete [] r0;
} // end omp parallel

    return result;

}

template <typename T>
void evolve(py::array_t<std::complex<T>, py::array::c_style> op, py::array_t<std::complex<T>, py::array::c_style> psi, const std::vector<int> &vdims, const unsigned short idx) {
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");

    if (idx < 0 || idx >= vdims.size())
        throw std::runtime_error("Index for permutation is invalid");


    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);


    size_t op_dim = op_buf.shape[0];
    size_t psi_dim = psi_buf.shape[0];
    size_t steps = psi_buf.shape[0]/op_buf.shape[0];


    // === indexing stuff ===
    size_t ndims = vdims.size();
    size_t dims[ndims];
    size_t prod[ndims];
    for (size_t i=0;i<ndims;i++) {
        dims[i]=vdims[i];
    }

    for (size_t i=0;i<ndims;i++){
        prod[i]=1;
        for (size_t j=i+1;j<ndims;j++)
            prod[i] *= dims[j];
    }

    // swap index to be permuted
    size_t tmp = prod[ndims-2];
    prod[ndims-2] = prod[idx];
    prod[idx] = tmp;

    // allocate and generate
    size_t * idxes = new size_t[psi_dim];
    gen_idxes(dims, prod, ndims, idxes);
    // === end of indexing stuff ===


#pragma omp parallel default(none) shared(steps,psi_ptr,op_ptr,idxes) firstprivate(op_dim)
{
    size_t s;
    std::complex<T> *r0 = new std::complex<T>[op_dim];
    for (size_t i = 0; i < op_dim; i++)
        r0[i] = {0.0,0.0};

#pragma omp for
    for (size_t step = 0; step < steps; step++) {
        s = step * op_dim;

        // dot product
        for (size_t i = 0; i < op_dim; i++) {
            for (size_t j = 0; j < op_dim; j++) {
                r0[i] += op_ptr[i*op_dim + j] * psi_ptr[idxes[s+j]];
            }
        }
        // set psi
        for (size_t i = 0; i < op_dim; i++) {
            psi_ptr[idxes[s+i]] = r0[i];
            // reset r0[i]
            r0[i] = {0.0,0.0};
        }
    }

    delete [] r0;
} // end omp parallel
    delete [] idxes;
}

PYBIND11_MODULE(fast_quantum, m) {
    m.def("measure", &measure<double>, "Compute observable appearing as the last operator in the Hilbert space");
    m.def("measure", &measure<float>, "Compute observable appearing as the last operator in the Hilbert space");
    m.def("evolve", &evolve<double>, "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");
    m.def("evolve", &evolve<float>, "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");
}
