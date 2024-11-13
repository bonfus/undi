#include "mpi_wrap.hpp"
// see here: https://github.com/mpi4py/mpi4py/issues/19#issuecomment-768143143
#ifdef MSMPI_VER
#define PyMPI_HAVE_MPI_Message 1
#endif
#include <mpi4py/mpi4py.h>

#include <iostream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>


#include "permutations.hpp"
#include "handler.hpp"

namespace py = pybind11;
using namespace std;

/*
 * Do not try to understand this code. Check the python implementation instead.
 * It's in celio_on_steroids.
 *
 */

/*! Return a MPI communicator from mpi4py communicator object. */
MPI_Comm *get_mpi_comm(py::object py_comm) {
  MPI_Comm *comm_ptr = PyMPIComm_Get(py_comm.ptr());

  if (!comm_ptr)
    throw py::error_already_set();

  return comm_ptr;
}

template <typename T>
void evolve_mpi(py::array_t<std::complex<T>, py::array::c_style> op,
                py::array_t<std::complex<T>, py::array::c_style> psi,
                py::array_t<size_t, py::array::c_style> dims, size_t spin,
                py::object py_comm, size_t batch_dim) {


    MPI_Comm comm = *get_mpi_comm(py_comm);

    int size = 0;
    MPI_Comm_size(comm, &size);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // get data from python
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();
    py::buffer_info dims_buf = dims.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");


    // collect pointers of psi and operator to be applied.
    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);


    // compute index - init part
    size_t nspins = dims_buf.shape[0];
    size_t *dims_ptr = static_cast<size_t *>(dims_buf.ptr);
    size_t *prods_ptr = new size_t[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;
    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    std::swap(dims_ptr[spin], dims_ptr[nspins-2]);
    std::swap(prods_ptr[spin], prods_ptr[nspins-2]);
    // end compute index - init part


    // compute range of global index of wfc
    size_t start = rank  * psi_buf.shape[0];
    size_t end   = start + psi_buf.shape[0];

    // compute dimensions of the operators and the block products to be computed
    size_t op_dim = op_buf.shape[0];
    size_t steps  = (psi_buf.shape[0]*size)/op_dim;

    // Use batch to speedup communication
    // check batch dim
    if (steps / batch_dim <= 0) throw std::invalid_argument("Batch size too large");
    if ((steps % batch_dim) != 0) throw std::invalid_argument("Invalid batch size");

    // skip comm when not needed
    if ((psi_buf.shape[0] % op_dim) != 0) throw std::invalid_argument("Invalid batch size");
    bool need_communication = (spin != nspins-2) || (size == 1);

    // prepare data
    steps = steps / batch_dim;

    // start (global index), batch (local index), lidx (local index of psi)
    size_t s, b, lidx;

    // array hosting the results of (batched) matrix vector multiplication
    std::complex<T> *r0 = new std::complex<T>[op_dim * batch_dim];

    // array hosting the local indices for a given permutation
    size_t *idxes_ptr = new size_t[op_dim * batch_dim];


#pragma omp parallel default(none) \
            shared(steps, psi_ptr, idxes_ptr, op_ptr, \
                   batch_dim, op_dim, r0, start, end, dims_ptr, \
                   prods_ptr, nspins, comm, cout) \
            private(s,b,lidx) firstprivate(need_communication)
{

    for (size_t step = 0; step < steps; step++) {

// WORKS
//        size_t idx_start[nspins];
//        size_t aux[nspins];
//        size_t idx=0;

//        s = step *  batch_dim * op_dim;
//        flat_to_nd_idx(s, idx_start, dims_ptr, prods_ptr, nspins);
//        iterate(0, nspins, dims_ptr, idx_start, aux, prods_ptr, &idx, idxes_ptr, op_dim * batch_dim);
// END WORKS


// ALSO WORKS
/*
#pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            size_t idx_start[nspins];
            size_t aux[nspins];
            size_t idx=0;

            b = batch_step * op_dim;
            s = step *  batch_dim * op_dim  + b;

            idx=0;
            flat_to_nd_idx(s, idx_start, dims_ptr, prods_ptr, nspins);
            iterate(0, nspins, dims_ptr, idx_start, aux, prods_ptr, &idx, &idxes_ptr[b], op_dim);
        }
*/
// END ALSO WORKS


// Another ATTEMPT (faster!!)
        // distribute data among threads allowing largest data parallelism
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t first = step *  batch_dim * op_dim;
        size_t last = first + batch_dim * op_dim;
        size_t tstep = (last-first) / nthreads;

        size_t chunk = tstep;
        if ( tid == nthreads - 1)
            chunk += (last-first) % nthreads;

        // compute indeces
        size_t idx_start[nspins];
        size_t aux[nspins];
        size_t idx=0;

        flat_to_nd_idx(first + tid*tstep, idx_start, dims_ptr, prods_ptr, nspins);
        iterate(0, nspins, dims_ptr, idx_start, aux, prods_ptr, &idx, &idxes_ptr[tid*tstep], chunk);

//pragma omp barrier (paranoid barrier)
/*
//pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;
            //cout << tid << " " << batch_step << endl;
            if ( idxes_ptr[b] >= end ) {
                for (size_t i = 0; i < op_dim; i++)
                    r0[i+b] = {0.0,0.0};
                continue;
            }

            // dot product
            for (size_t i = 0; i < op_dim; i++) {
                // reset r0[i]
                r0[i+b] = {0.0,0.0};
                for (size_t j = 0; j < op_dim; j++) {

                    // check if compute moving to local idx
                    lidx = idxes_ptr[b+j];
                    if (lidx < start) continue;
                    if (lidx >= end) continue;
                    //lidx = lidx - start

                    r0[i+b] += op_ptr[i*op_dim + j] * psi_ptr[lidx - start];
                }
            }
        }
*/
#pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;
            //cout << tid << " " << batch_step << endl;

            for (size_t i = 0; i < op_dim; i++)
                r0[i+b] = {0.0,0.0};

            if ( idxes_ptr[b] >= end ) continue;

            // dot product
            for (size_t j = 0; j < op_dim; j++) {
                lidx = idxes_ptr[b+j];
                if (lidx < start) continue;
                if (lidx >= end) continue;
                lidx -= start;

                for (size_t i = 0; i < op_dim; i++) {
                    // check if compute moving to local idx
                    r0[i+b] += op_ptr[i*op_dim + j] * psi_ptr[lidx];
                }
            }
        }
        if (need_communication) {
#pragma omp master
{
            mp::sum(r0, op_dim * batch_dim, comm); // no error checking here, maybe not wise.

} // end omp master
#pragma omp barrier
        }


        // update psi
{
#pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;
            if ( idxes_ptr[b] >= end ) continue;

            for (size_t i = 0; i < op_dim; i++) {
                lidx = idxes_ptr[b+i];

                if (lidx < start) continue;
                if (lidx >= end) continue;
                lidx -= start;

                psi_ptr[lidx] = r0[b+i];
            }
        }
} // omp for
    }
} //omp
    delete [] r0;

    // delete index stuff
    delete [] idxes_ptr; delete [] prods_ptr;
}


template <typename T>
void evolve_mpiBACKUP(py::array_t<std::complex<T>, py::array::c_style> op,
                py::array_t<std::complex<T>, py::array::c_style> psi,
                py::array_t<size_t, py::array::c_style> dims, size_t spin,
                py::object py_comm, size_t batch_dim) {


    MPI_Comm comm = *get_mpi_comm(py_comm);

    int size = 0;
    MPI_Comm_size(comm, &size);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // get data from python
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();
    py::buffer_info dims_buf = dims.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");


    // collect pointers of psi and operator to be applied.
    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);


    // compute index - init part
    size_t nspins = dims_buf.shape[0];
    size_t *dims_ptr = static_cast<size_t *>(dims_buf.ptr);
    size_t *prods_ptr = new size_t[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;
    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    std::swap(dims_ptr[spin], dims_ptr[nspins-2]);
    std::swap(prods_ptr[spin], prods_ptr[nspins-2]);
    // end compute index - init part


    // compute range of global index of wfc
    size_t start = rank  * psi_buf.shape[0];
    size_t end   = start + psi_buf.shape[0];

    // compute dimensions of the operators and the block products to be computed
    size_t op_dim = op_buf.shape[0];
    size_t steps  = (psi_buf.shape[0]*size)/op_dim;


    assert(SIZE_MAX > psi_buf.shape[0]*size); // SIZE_MAX is used to identify not found elements

    // Use batch to speedup communication
    // check batch dim
    if (steps / batch_dim <= 0) throw std::invalid_argument("Batch size too large");
    if ((steps % batch_dim) != 0) throw std::invalid_argument("Invalid batch size");

    // skip comm when not needed
    if ((psi_buf.shape[0] % op_dim) != 0) throw std::invalid_argument("Invalid batch size");
    bool need_communication = (spin != nspins-2);

    // prepare data
    steps = steps / batch_dim;

    // start (global index), batch (local index), lidx (local index of psi)
    size_t s, b, lidx;

    // array hosting the results of (batched) matrix vector multiplication
    std::complex<T> *r0 = new std::complex<T>[op_dim * batch_dim];

    // array hosting the local indices for a given permutation
    size_t *idxes_ptr = new size_t[op_dim * batch_dim];


#pragma omp parallel default(none) \
            shared(steps, psi_ptr, idxes_ptr, op_ptr, \
                   batch_dim, op_dim, r0, start, end, dims_ptr, \
                   prods_ptr, nspins, comm, cout) \
            private(s,b,lidx) firstprivate(need_communication)
{

    for (size_t step = 0; step < steps; step++) {

// WORKS
//        size_t idx_start[nspins];
//        size_t aux[nspins];
//        size_t idx=0;

//        s = step *  batch_dim * op_dim;
//        flat_to_nd_idx(s, idx_start, dims_ptr, prods_ptr, nspins);
//        iterate(0, nspins, dims_ptr, idx_start, aux, prods_ptr, &idx, idxes_ptr, op_dim * batch_dim);
// END WORKS


// ALSO WORKS
/*
#pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            size_t idx_start[nspins];
            size_t aux[nspins];
            size_t idx=0;

            b = batch_step * op_dim;
            s = step *  batch_dim * op_dim  + b;

            idx=0;
            flat_to_nd_idx(s, idx_start, dims_ptr, prods_ptr, nspins);
            iterate(0, nspins, dims_ptr, idx_start, aux, prods_ptr, &idx, &idxes_ptr[b], op_dim);
        }
*/
// END ALSO WORKS


// Another ATTEMPT (faster!!)
        // distribute data among threads allowing largest data parallelism
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t first = step *  batch_dim * op_dim;
        size_t last = first + batch_dim * op_dim;
        size_t tstep = (last-first) / nthreads;

        size_t chunk = tstep;
        if ( tid == nthreads - 1)
            chunk += (last-first) % nthreads;

        // compute indeces
        size_t idx_start[nspins];
        size_t aux[nspins];
        size_t idx=0;

        flat_to_nd_idx(first + tid*tstep, idx_start, dims_ptr, prods_ptr, nspins);
        iterate(0, nspins, dims_ptr, idx_start, aux, prods_ptr, &idx, &idxes_ptr[tid*tstep], chunk);

//pragma omp barrier (paranoid barrier)

#pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;
            //cout << tid << " " << batch_step << endl;
            if ( idxes_ptr[b] >= end ) {
                for (size_t i = 0; i < op_dim; i++)
                    r0[i+b] = {0.0,0.0};
                continue;
            }

            // dot product
            for (size_t i = 0; i < op_dim; i++) {
                // reset r0[i]
                r0[i+b] = {0.0,0.0};
                for (size_t j = 0; j < op_dim; j++) {

                    // check if compute moving to local idx
                    lidx = idxes_ptr[b+j];
                    if (lidx < start) continue;
                    if (lidx >= end) continue;
                    //lidx = lidx - start

                    r0[i+b] += op_ptr[i*op_dim + j] * psi_ptr[lidx - start];
                }
            }
        }
        if (need_communication) {
#pragma omp master
{
            mp::sum(r0, op_dim * batch_dim, comm); // no error checking here, maybe not wise.

} // end omp master
#pragma omp barrier
        }


        // update psi
{
#pragma omp for schedule(static)
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;
            if ( idxes_ptr[b] >= end ) continue;

            for (size_t i = 0; i < op_dim; i++) {
                lidx = idxes_ptr[b+i];

                if (lidx < start) continue;
                if (lidx >= end) continue;
                //lidx = lidx - start;

                psi_ptr[lidx - start] = r0[b+i];
            }
        }
} // omp for
    }
} //omp
    delete [] r0;

    // delete index stuff
    delete [] idxes_ptr; delete [] prods_ptr;
}





template <typename T>
void evolve_mpi2(py::array_t<std::complex<T>, py::array::c_style> op,
                py::array_t<std::complex<T>, py::array::c_style> psi,
                py::array_t<size_t, py::array::c_style> dims, size_t spin,
                py::object py_comm, size_t batch_dim) {


    MPI_Comm comm = *get_mpi_comm(py_comm);
    int size;
    MPI_Comm_size(comm, &size);
    int rank;
    MPI_Comm_rank(comm, &rank);

    //if (rank == 0) cout << "called evolve_mpi2 with spin " << spin << endl;
    // get data from python
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();
    py::buffer_info dims_buf = dims.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");


    // collect pointers of psi and operator to be applied.
    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);


    // compute index - init part
    size_t nspins = dims_buf.shape[0];
    size_t *dims_ptr = static_cast<size_t *>(dims_buf.ptr);
    size_t *prods_ptr = new size_t[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;
    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    std::swap(dims_ptr[spin], dims_ptr[nspins-2]);
    std::swap(prods_ptr[spin], prods_ptr[nspins-2]);
    // end compute index - init part


    // compute range of global index of wfc
    size_t wfc_size = psi_buf.shape[0];
    size_t start = rank  * wfc_size;
    size_t end   = start + wfc_size;

    // compute dimensions of the operators and the block products to be computed
    size_t op_dim = op_buf.shape[0];
    size_t steps  = (psi_buf.shape[0]*size)/op_dim;
    size_t spin_dim = op_dim / 2;

    // Use batch to speedup communication
    // check batch dim
    if (steps / batch_dim <= 0) throw std::invalid_argument("Batch size too large");
    if ((steps % batch_dim) != 0) throw std::invalid_argument("Invalid batch size");

    // skip comm when not needed
    if ((psi_buf.shape[0] % op_dim) != 0) throw std::invalid_argument("Invalid batch size");
    bool need_communication = (spin != nspins-2) && (size != 1);

    // prepare data
    steps = steps / batch_dim;

    // start (global index), batch (local index), lidx (local index of psi)
    size_t s, b, lidx;


    // array hosting the results of (batched) matrix vector multiplication
    std::complex<T> *r0 = new std::complex<T>[op_dim * batch_dim];

    // check we can evenly split subbatches among threads
    if ( batch_dim % omp_get_max_threads() != 0 ) throw std::invalid_argument("Invalid number of threads!");

#pragma omp parallel default(none) \
            shared(steps, psi_ptr, op_ptr, \
                   batch_dim, op_dim, r0, start, end, dims_ptr, \
                   prods_ptr, nspins, comm, spin_dim) \
            private(s,b,lidx) firstprivate(need_communication)
{

    // distribute data among threads allowing largest data parallelism
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    // parallel over threads
    size_t subbatch_dim = batch_dim / nthreads;

    size_t subbatch_start = tid * subbatch_dim;
    size_t subbatch_end = subbatch_start + subbatch_dim;

    size_t idx_nd[nspins]; // n-dimensional index
    size_t idx=0;          // linear index

    size_t chunk_dim = subbatch_dim * op_dim;
    size_t idxes_ptr[chunk_dim];
    size_t first = 0;

    for (size_t step = 0; step < steps; step++)
    {

        // compute indeces
        flat_to_nd_idx(first + subbatch_start*op_dim, idx_nd, dims_ptr, prods_ptr, nspins);
        first += batch_dim * op_dim;

        // reset
#pragma omp barrier
        idx=0;
        for (size_t i = subbatch_start*op_dim; i < op_dim * subbatch_end; i++)
        {
            r0[i] = {0.0, 0.0};
        }

        for (size_t batch_step = subbatch_start; batch_step < subbatch_end; batch_step++) {

            b = batch_step * op_dim;

            for (size_t j = 0; j < spin_dim; j++) {

                lidx = fill(idx_nd, prods_ptr, nspins);

                if ( lidx < start || lidx >= end ){
                    iterate2(nspins, dims_ptr, idx_nd);
                    iterate2(nspins, dims_ptr, idx_nd);
                    continue;
                }
                // move to local index
                lidx -= start;

                // second last is the "active" one
                s = 2 * idx_nd[nspins-2];

                // compute for later
                idxes_ptr[idx] = lidx;
                idxes_ptr[idx+1] = s + b;
                idx+=2;

                // dot product
                for (size_t i = 0; i < op_dim; i++) {
                    r0[i + b] += op_ptr[i*op_dim + s] * psi_ptr[lidx];
                    r0[i + b] += op_ptr[i*op_dim + s + 1] * psi_ptr[lidx + 1];
                }
                iterate2(nspins, dims_ptr, idx_nd);
                iterate2(nspins, dims_ptr, idx_nd);
            }

        }

        if (need_communication) {
#pragma omp barrier
#pragma omp master
            mp::sum(r0, op_dim * batch_dim, comm); // no error checking here, maybe not wise.
#pragma omp barrier
        }


        // update psi
        for (size_t i = 0; i < idx; i+=2) {
            lidx = idxes_ptr[i];
            s = idxes_ptr[i+1];
            psi_ptr[lidx] = r0[s];
            psi_ptr[lidx+1] = r0[s+1];
        }
    }
} //omp
    delete [] r0;

    // delete index stuff
    delete [] prods_ptr;
}
































































































/*



template <typename T>
void evolve_mpi3(py::array_t<std::complex<T>, py::array::c_style> op,
                py::array_t<std::complex<T>, py::array::c_style> psi,
                py::array_t<size_t, py::array::c_style> dims, size_t spin,
                py::object py_comm, size_t batch_dim) {


    MPI_Comm comm = *get_mpi_comm(py_comm);
    int size;
    MPI_Comm_size(comm, &size);
    int rank;
    MPI_Comm_rank(comm, &rank);

    //if (rank == 0) cout << "called evolve_mpi2 with spin " << spin << endl;
    // get data from python
    py::buffer_info op_buf = op.request(), psi_buf = psi.request();
    py::buffer_info dims_buf = dims.request();

    if (op_buf.ndim != 2 || psi_buf.ndim != 1)
        throw std::runtime_error("Operator must be a 2D array, wavefunction a 1D array.");

    if ((psi_buf.shape[0] % op_buf.shape[0]) != 0)
        throw std::runtime_error("Something is very wrong.");


    // collect pointers of psi and operator to be applied.
    complex<T> *op_ptr = static_cast<complex<T> *>(op_buf.ptr);
    complex<T> *psi_ptr = static_cast<complex<T> *>(psi_buf.ptr);


    // compute index - init part
    size_t nspins = dims_buf.shape[0];
    size_t *dims_ptr = static_cast<size_t *>(dims_buf.ptr);
    size_t *prods_ptr = new size_t[nspins];

    for (size_t i = 0; i < nspins; i++)
        prods_ptr[i] = 1;
    for (size_t i = 0; i < nspins; i++) {
        for (size_t j = i+1; j < nspins; j++) {
            prods_ptr[i] *= dims_ptr[j];
        }
    }

    std::swap(dims_ptr[spin], dims_ptr[nspins-2]);
    std::swap(prods_ptr[spin], prods_ptr[nspins-2]);
    // end compute index - init part


    // compute range of global index of wfc
    size_t wfc_size = psi_buf.shape[0];
    size_t start = rank  * wfc_size;
    size_t end   = start + wfc_size;

    // compute dimensions of the operators and the block products to be computed
    size_t op_dim = op_buf.shape[0];
    size_t steps  = (psi_buf.shape[0]*size)/op_dim;
    size_t spin_dim = op_dim / 2;

    // Use batch to speedup communication
    // check batch dim
    if (steps / batch_dim <= 0) throw std::invalid_argument("Batch size too large");
    if ((steps % batch_dim) != 0) throw std::invalid_argument("Invalid batch size");

    // skip comm when not needed
    if ((psi_buf.shape[0] % op_dim) != 0) throw std::invalid_argument("Invalid batch size");
    bool need_communication = (spin != nspins-2) && (size != 1);

    // prepare data
    steps = steps / batch_dim;

    // start (global index), batch (local index), lidx (local index of psi)
    size_t s, b, lidx;


    // array hosting the results of (batched) matrix vector multiplication
    std::complex<T> *r0 = new std::complex<T>[size, op_dim * batch_dim];

    // check we can evenly split subbatches among threads
    if ( batch_dim % omp_get_max_threads() != 0 ) throw std::invalid_argument("Invalid number of threads!");

#pragma omp parallel default(none) \
            shared(steps, psi_ptr, op_ptr, \
                   batch_dim, op_dim, r0, start, end, dims_ptr, \
                   prods_ptr, nspins, comm, spin_dim) \
            private(s,b,lidx) firstprivate(need_communication)
{

    // distribute data among threads allowing largest data parallelism
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    // parallel over threads
    size_t subbatch_dim = batch_dim / nthreads;

    size_t subbatch_start = tid * subbatch_dim;
    size_t subbatch_end = subbatch_start + subbatch_dim;

    size_t idx_nd[nspins]; // n-dimensional index
    size_t idx[size]=0;          // linear index

    size_t chunk_dim = subbatch_dim * op_dim;
    size_t idxes_ptr[size][chunk_dim];
    size_t first = 0;
    size_t impi;

    for (size_t step = 0; step < steps; step++) {

        // compute indeces
        flat_to_nd_idx(first + subbatch_start*op_dim, idx_nd, dims_ptr, prods_ptr, nspins);
        first += batch_dim * op_dim;

        // reset
#pragma omp barrier
        for ( impi = 0; impi < size; impi++)
            idx[impi]=0;

        for (size_t i = subbatch_start*op_dim; i < op_dim * subbatch_end; i++)
        {
            r0[i] = {0.0, 0.0};
        }

        for (size_t batch_step = subbatch_start; batch_step < subbatch_end; batch_step++) {

            b = batch_step * op_dim;

            // compute all indeces fot the next 8 elemenys
            for (size_t i = 0; i < op_dim; i++) {

                lidx = fill(idx_nd, prods_ptr, nspins);
                s = 2 * idx_nd[nspins-2];

                // who own this?
                for ( impi = 0; impi < size; impi++) {
                    if ( lidx < (impi+1)*wfc_size) {
                        // where to get the element
                        idxes_ptr[impi][idx[impi]] = lidx;
                        // where in the matrix
                        idxes_ptr[impi][idx[impi]+1] = s;
                        idx[impi] = idx[impi] + 2;
                        break;
                    }
                }
                iterate2(nspins, dims_ptr, idx_nd);
            }



            for (size_t i = 0; i < op_dim; i++) {
                // reset r0[i]
                r0[i+b] = {0.0,0.0};
                for (size_t j = 0; j < op_dim; j++) {

                    // check if compute moving to local idx
                    lidx = idxes_ptr[b+j];
                    if (lidx < start) continue;
                    if (lidx >= end) continue;
                    //lidx = lidx - start

                    r0[i+b] += op_ptr[i*op_dim + j] * psi_ptr[lidx - start];
                }
            }



            for (size_t j = 0; j < spin_dim; j++) {

                //lidx = fill(idx_nd, prods_ptr, nspins);
                lidx = idx_nd

                // move to local index
                lidx -= start;

                // second last is the "active" one
                s = 2 * idx_nd[nspins-2];

                // compute for later
                idxes_ptr[idx] = lidx;
                idxes_ptr[idx+1] = s + b;
                idx+=2;

                // dot product
                for (size_t i = 0; i < op_dim; i++) {
                    r0[impi][i + b] += op_ptr[i*op_dim + s] * psi_ptr[lidx];
                    r0[impi][i + b] += op_ptr[i*op_dim + s + 1] * psi_ptr[lidx + 1];
                }
                iterate2(nspins, dims_ptr, idx_nd);
                iterate2(nspins, dims_ptr, idx_nd);
            }

        }

        if (need_communication) {
#pragma omp barrier
#pragma omp master
            mp::sum(r0, op_dim * batch_dim, comm); // no error checking here, maybe not wise.
#pragma omp barrier
        }


        // update psi
        for (size_t i = 0; i < idx; i+=2) {
            lidx = idxes_ptr[i];
            s = idxes_ptr[i+1];
            psi_ptr[lidx] = r0[s];
            psi_ptr[lidx+1] = r0[s+1];
        }
    }
} //omp
    delete [] r0;

    // delete index stuff
    delete [] prods_ptr;
}

*/


template <typename T>
double measure_mpi(py::array_t<std::complex<T>, py::array::c_style> op,
                   py::array_t<std::complex<T>, py::array::c_style> psi,
                   py::array_t<std::complex<T>, py::array::c_style> aux,
                   Handler &h) {


    MPI_Comm comm = h.comm; //*get_mpi_comm(py_comm);

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

    if ( mp::sum(&result, 1, comm) !=  MPI_SUCCESS)
        throw std::runtime_error("MPI failed.");

    return result;

}


PYBIND11_MODULE(fast_quantum_mpi, m) {

    // initialize mpi4py's C-API
    if (import_mpi4py() < 0) {
      // mpi4py calls the Python C API, we let pybind11 give us the detailed traceback
      throw py::error_already_set();
    }

    //m.def("initialize", &init_idx, "Initialize indeces");
    //m.def("initialize_mpi", &init_mpi, "Initialize MPI");
    //m.def("finalize",  &finalize, "Clear data");

    m.def("measure_mpi", &measure_mpi<double>, "Compute observable appearing as the last operator in the Hilbert space");
    m.def("measure_mpi", &measure_mpi<float>, "Compute observable appearing as the last operator in the Hilbert space");
    m.def("evolve_mpi", &evolve_mpi2<double>, "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");
    m.def("evolve_mpi", &evolve_mpi2<float>, "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");

}
