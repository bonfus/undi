#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <omp.h>

#if defined(__MKL)
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#endif

#include <iostream>
#include "permutations.hpp"
#include "handler.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

namespace py = pybind11;
using namespace std;

/*
 * Do not try to understand this code. Check the python implementation instead.
 * It's in celio_on_steroids.
 *
 */


void initialize(py::array_t<size_t, py::array::c_style> dims, Handler &h) {

    py::buffer_info  dims_buf = dims.request();

    if (dims_buf.ndim != 1)
        throw std::runtime_error("Dims must be 1D array.");

    size_t *dims_ptr = static_cast<size_t *>(dims_buf.ptr);
    size_t nspins = h.nspins = dims_buf.shape[0];


    // compute Hilbert space dimension
    size_t hdim = hilbert_space_dims(nspins, dims_ptr);

    // save info to handler
    h.h_dim = hdim;
    h.nspins = nspins;

    // mem footprint can be smaller!
    size_t *idx_ptr = new size_t[hdim * nspins];
    size_t *inv_idx_ptr = new size_t[hdim * nspins];


    for (size_t i = 0; i < nspins; i++) {
        compute_idx_invidx(hdim, dims_ptr, nspins, i, &idx_ptr[i * hdim], &inv_idx_ptr[i * hdim]);
    }

    //h.rotations = new size_t[hdim * (nspins-1)];
    h.inv_rotations = new size_t[hdim * (nspins-1)];

    for (size_t j = 0; j < hdim; j++)
        //h.rotations[j] = idx_ptr[j];  // where to look for elements
        h.inv_rotations[idx_ptr[j]] = j;    // where to "send" elements

    size_t s;
    size_t ps;
    for (size_t i = 1; i < nspins-1; i++) {
        s = i*hdim;
        ps = (i-1)*hdim;
        for (size_t j = 0; j < hdim; j++) {
            //h.rotations[i*hdim + j] = inv_idx_ptr[ (i-1)*hdim + idx_ptr[i*hdim + j] ];
            h.inv_rotations[s + inv_idx_ptr[ ps + idx_ptr[s + j] ]] = j;
        }
    }

    delete [] idx_ptr;
    delete [] inv_idx_ptr;

#if defined(__XSMM)
    // create xsmm kernels
#error not implemented
#endif

    h.initialized = true;

}

void finalize(Handler &h) {
    //delete [] h.rotations;
    delete [] h.inv_rotations;
    h.initialized = false;
}



template <typename T>
void evolve(py::array_t<std::complex<T>, py::array::c_style> op,
            py::array_t<std::complex<T>, py::array::c_style> psi,
            py::array_t<std::complex<T>, py::array::c_style> aux,
            size_t spin, Handler &h) {
    py::buffer_info op_buf = op.request(),
                   psi_buf = psi.request(),
                   aux_buf = aux.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");


    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);
    complex<T> *aux_ptr = static_cast<complex<T> *>(aux_buf.ptr);

    size_t *inv_rotations_ptr = h.inv_rotations;

    size_t wfc_dim = psi_buf.shape[0];
    size_t op_dim = op_buf.shape[0];

    size_t steps = wfc_dim/op_dim;

    // reorder for new operation, s points to rotation required to obtain
    // the permutation for this spin from the previous one.
    size_t s = spin*wfc_dim;


#if defined(__MKL)

    if ( steps < omp_get_max_threads() ) throw std::invalid_argument("Too many threads!"); // TODO: fix more threads than steps!

#pragma omp parallel default(none) shared(psi_ptr,op_ptr,aux_ptr,inv_rotations_ptr) firstprivate(op_dim,wfc_dim) firstprivate(steps, s)
{
    const std::complex<T> alpha = {1.0, 0.0};
    const std::complex<T> beta = {0.0, 0.0};

    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    // fix if more threads than steps!
    steps = steps / nthreads;
    if ( tid == nthreads - 1)
        steps += steps % nthreads;


#pragma omp for schedule(static)
    for (size_t i = 0; i < wfc_dim; i++)
        aux_ptr[inv_rotations_ptr[s + i]] = psi_ptr[i];
        //aux_ptr[i] = psi_ptr[rotations_ptr[s + i]];

#pragma omp barrier
    // use float or double here
    if (typeid(T) == typeid(double)) {
        cblas_zgemv_batch_strided(CblasRowMajor, CblasNoTrans,
                                  op_dim, op_dim, &alpha,
                                  op_ptr, op_dim, 0,
//                                  aux_ptr, 1, op_dim, &beta,
//                                  psi_ptr, 1, op_dim, steps);
                                  &aux_ptr[op_dim*steps*tid], 1, op_dim, &beta,
                                  &psi_ptr[op_dim*steps*tid], 1, op_dim, steps);
   } else {
        cblas_cgemv_batch_strided(CblasRowMajor, CblasNoTrans,
                                  op_dim, op_dim, &alpha,
                                  op_ptr, op_dim, 0,
//                                  aux_ptr, 1, op_dim, &beta,
//                                  psi_ptr, 1, op_dim, steps);
                                  &aux_ptr[op_dim*steps*tid], 1, op_dim, &beta,
                                  &psi_ptr[op_dim*steps*tid], 1, op_dim, steps);
   }
} // end omp parallel

#else

    size_t os;

    // make me parallel!
    for (size_t i = 0; i < wfc_dim; i++)
        //aux_ptr[i] = psi_ptr[rotations_ptr[s + i]];
        aux_ptr[inv_rotations_ptr[s + i]] = psi_ptr[i];


#pragma omp parallel default(none) shared(steps,psi_ptr,op_ptr,aux_ptr) firstprivate(op_dim) private(s, os)
{
#pragma omp for schedule(static)
    for (size_t step = 0; step < steps; step++) {
        s = step * op_dim;

        // dot product
        for (size_t i = 0; i < op_dim; i++) {
            // reset new_psi element
            os = i*op_dim;
            psi_ptr[i+s] = op_ptr[os] * aux_ptr[s];

            for (size_t j = 1; j < op_dim; j++) {
                psi_ptr[i+s] += op_ptr[os + j] * aux_ptr[s+j];
            }
        }
    }
} // end omp parallel
#endif
}



template <typename T>
double measure(py::array_t<std::complex<T>, py::array::c_style> op,
               py::array_t<std::complex<T>, py::array::c_style> psi,
               py::array_t<std::complex<T>, py::array::c_style> aux,
               Handler &h) {


    py::buffer_info op_buf = op.request(),
                   psi_buf = psi.request(),
                   aux_buf = aux.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong");

    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);
    complex<T> *aux_ptr = static_cast<complex<T> *>(aux_buf.ptr);

    size_t wfc_dim = psi_buf.shape[0];
    size_t op_dim = op_buf.shape[0];
    size_t steps = wfc_dim/op_dim;
    double result = 0.0;

#if defined(__MKL_THANKS_BUT_NO)
// this is slower!

    const std::complex<T> alpha = {1.0, 0.0} ;
    const std::complex<T> beta = {0.0, 0.0};
    std::complex<T> r = 0.0;

    // use float or double here
    cblas_zgemv_batch_strided(CblasRowMajor, CblasNoTrans,
                                op_dim, op_dim, &alpha,
                                op_ptr, op_dim, 0,
                                psi_ptr, 1, op_dim, &beta,
                                aux_ptr, 1, op_dim, steps);

    //cblas_zdotc (&wfc_dim, psi_ptr, &uno, aux_ptr, &uno, &r);
    cblas_zdotc_sub( wfc_dim, psi_ptr, 1, aux_ptr, 1, &r );
    result = r.real();
#else

    // makes compiler happy, but it's used above
    UNUSED(aux_ptr);

#pragma omp parallel reduction(+:result) default(none) shared(steps,psi_ptr,op_ptr) firstprivate(op_dim)
{
    size_t s;
    std::complex<T> *r0 = new std::complex<T>[op_dim];

    // loop on slices
#pragma omp for schedule(static)
    for (size_t step = 0; step < steps; step++) {
        s = step * op_dim;
        // dot product
        for (size_t i = 0; i < op_dim; i++) {
            r0[i] = {0.0,0.0};
            for (size_t j = 0; j < op_dim; j++) {
                r0[i] += op_ptr[i*op_dim + j] * psi_ptr[s+j];
            }
        }

        for (size_t i = 0; i < op_dim; i++) {
            result += ( conj(psi_ptr[s+i]) * r0[i] ).real();
        }
    }
    delete [] r0;
} // end omp parallel
#endif

    return result;

}

PYBIND11_MODULE(fast_quantum_blas, m) {
    py::class_<Handler>(m, "Handler", py::module_local())
    .def(py::init<>())
    .def_readonly("initialized", &Handler::initialized);

    m.def("initialize", &initialize, "Initialize index computation");
    m.def("finalize", &finalize, "Free up resources");
    m.def("measure", &measure<double>,
            py::arg("op").noconvert(), py::arg("psi").noconvert(), py::arg("aux").noconvert(),
            py::arg("h"),
            "Compute observable appearing as the last operator in the Hilbert space");
    m.def("measure", &measure<float>,
            py::arg("op").noconvert(), py::arg("psi").noconvert(), py::arg("aux").noconvert(),
            py::arg("h"),
            "Compute observable appearing as the last operator in the Hilbert space");
    m.def("evolve", &evolve<double>,
           py::arg("op").noconvert(), py::arg("psi").noconvert(), py::arg("aux").noconvert(),
           py::arg("spin"), py::arg("h"),
            "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");
    m.def("evolve", &evolve<float>,
            py::arg("op").noconvert(), py::arg("psi").noconvert(), py::arg("aux").noconvert(),
            py::arg("spin"), py::arg("h"),
            "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");
}
