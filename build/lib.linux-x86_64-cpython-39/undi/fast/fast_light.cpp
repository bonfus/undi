#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include<iostream>
#include "handler.hpp"
#include "permutations.hpp"
#include <omp.h>

namespace py = pybind11;
using namespace std;

/*
 * Do not try to understand this code. Check the python implementation instead.
 * It's in celio_on_steroids.
 *
 */

void initialize(py::array_t<size_t, py::array::c_style> dims, Handler &h) {

    py::buffer_info dims_buf = dims.request();
    //, rotations_buf = rotations.request();

    if (dims_buf.ndim != 1)
        throw std::runtime_error("Dims must be 1D array.");

    size_t *dims_ptr = static_cast<size_t *>(dims_buf.ptr);
    size_t nspins = dims_buf.shape[0];

    // use fiill as usual
    size_t hdim = hilbert_space_dims(nspins, dims_ptr);
    h.h_dim = hdim;
    h.nspins = nspins;
    h.dims_ptr = new size_t[nspins];
    for (size_t i=0; i< nspins; i++)
        h.dims_ptr[i] = dims_ptr[i];

    h.initialized = true;
}

void finalize(Handler &h) {
    delete [] h.dims_ptr;
    h.initialized = false;
}

template <typename T>
double measure(py::array_t<std::complex<T>, py::array::c_style> op,
                py::array_t<std::complex<T>, py::array::c_style> psi,
                py::array_t<std::complex<T>, py::array::c_style> aux,
                Handler &h) {
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

    return result;

}

template <typename T>
void evolve(py::array_t<std::complex<T>, py::array::c_style> op,
            py::array_t<std::complex<T>, py::array::c_style> psi,
            py::array_t<std::complex<T>, py::array::c_style> aux,
            size_t spin,
            Handler &h) {
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");


    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);

    size_t op_dim = op_buf.shape[0];
    size_t steps = psi_buf.shape[0]/op_buf.shape[0];

    size_t nspins = h.nspins;
    size_t *dims_ptr = h.dims_ptr;
    size_t prods_ptr[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;

    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= h.dims_ptr[j];
        }
    }

    swap(dims_ptr, spin, nspins-2);
    swap(prods_ptr, spin, nspins-2);



#pragma omp parallel default(none) shared(steps,psi_ptr,op_ptr,nspins,dims_ptr,prods_ptr) \
                                    firstprivate(op_dim)
{
    std::complex<T> *r0 = new std::complex<T>[op_dim];
    size_t idxes_ptr[op_dim];
    size_t idx_nd[nspins];

    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    size_t chunk = steps / nthreads;
    size_t start = chunk * tid;
    size_t end = start + chunk;

    if ( tid == nthreads - 1)
        end += steps % nthreads;

    // compute starting value
    flat_to_nd_idx(start * op_dim, idx_nd, dims_ptr, prods_ptr, nspins);


    for (size_t step = start; step < end; step++) {

        for (size_t j = 0; j < op_dim; j++) {
            idxes_ptr[j] = fill(idx_nd, prods_ptr, nspins);
            iterate2(nspins, dims_ptr, idx_nd);
        }

        // dot product
        for (size_t i = 0; i < op_dim; i++) {
            // reset r0[i]
            r0[i] = {0.0,0.0};

            for (size_t j = 0; j < op_dim; j++) {
                r0[i] += op_ptr[i*op_dim + j] * psi_ptr[idxes_ptr[j]];
            }
        }
        // set psi
        for (size_t i = 0; i < op_dim; i++) {
            psi_ptr[idxes_ptr[i]] = r0[i];
        }

    }
    delete [] r0;
} // end omp parallel

    swap(dims_ptr, spin, nspins-2);
}

PYBIND11_MODULE(fast_quantum_light, m) {
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
