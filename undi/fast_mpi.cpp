#include <mpi.h>
// see here: https://github.com/mpi4py/mpi4py/issues/19#issuecomment-768143143
#ifdef MSMPI_VER
#define PyMPI_HAVE_MPI_Message 1
#endif
#include <mpi4py/mpi4py.h>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace py = pybind11;
using namespace std;

/*
 * Do not try to understand this code. Check the python implementation instead.
 * It's in celio_on_steroids.
 *
 */


/* MPI UTILS */
namespace mp { // Namespace for convenience

    template <typename T>
    MPI_Datatype get_type();

    template <>
    MPI_Datatype get_type<double>()
    {
        return MPI_DOUBLE;
    }

    template <>
    MPI_Datatype get_type<float>()
    {
        return MPI_FLOAT;
    }

    template <>
    MPI_Datatype get_type<std::complex<double>>()
    {
        return MPI_DOUBLE_COMPLEX;
    }

    template <>
    MPI_Datatype get_type<std::complex<float>>()
    {
        return MPI_COMPLEX;
    }

    template <typename T>
    int sum(T *buf, int count, const MPI_Comm comm)
    {
        return MPI_Allreduce(MPI_IN_PLACE, buf, count,
                                mp::get_type<T>(), MPI_SUM, comm);
    }

}


/*! Return a MPI communicator from mpi4py communicator object. */
MPI_Comm *get_mpi_comm(py::object py_comm) {
  MPI_Comm *comm_ptr = PyMPIComm_Get(py_comm.ptr());

  if (!comm_ptr)
    throw py::error_already_set();

  return comm_ptr;
}

/* Compute map from global to local index if global index is in my portion
 * of psi. This is set in lidx and true is returned. Otherwise false
 * is returned and lidx is not set.
 */
size_t g2l(size_t idx, size_t s, size_t e) {
    if (idx < s) return SIZE_MAX;
    if (idx >= e) return SIZE_MAX;

    return idx - s;
}
/* END MPI UTILS */



/* ===  Compute indexes and permutaions === */

/* NB: the implementation below is totally unreadable
 * The point of the entire code below is to perform this
 *
 * s = np.prod(dims)
 * np.swapaxes(np.arange(s,dtype=np.uint64).reshape(dims),spin,-2).flatten()
 *
 * where spin is the id of the spin under consideration and dims are the
 * dimensions of the subspaces forming the Hilber space.
 *
 * The math behind this is the reordering of numbers in represented with
 * variable base digits.
 * See
 */



size_t prod(size_t* v, size_t S, size_t nb) {
  size_t total = 1;
  for(size_t i=0; i<nb; i++) {
    if (i < S + 1) continue;
    total *= v[i];
  }
  return total;
}

size_t permute_idx(size_t n, size_t * pdims, size_t * perm, size_t * prods, size_t nb ){
    size_t res = 0;
    size_t c;

    for(size_t i = nb; i > 0;) {
        --i;

        c = n % pdims[i];
        n = n / pdims[i];

        res += c * prods[perm[i]];
    }
    return res;
}


void init_permutations(size_t * dims, size_t spin, size_t * prods, size_t * perm, size_t nb){

    // notice that prods are computed before being swapped.
    for (size_t i = 0; i < nb; i++)
        prods[i] = prod(dims, i, nb);

    for(size_t i = 0; i < nb;i++) {
        perm[i] = i;
    }

    std::swap(dims[spin], dims[nb-2]);
    std::swap(perm[spin], perm[nb-2]);
}



/* End compute indexes and permutaions */


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
    size_t *newod_ptr = new size_t[nspins];
    init_permutations(dims_ptr, spin, prods_ptr, newod_ptr, nspins);


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
    steps = steps / batch_dim;

    // start (global index), batch (local index), lidx (local index of psi)
    size_t s, b, lidx;

    // array hosting the results of (batched) matrix vector multiplication
    std::complex<T> *r0 = new std::complex<T>[op_dim * batch_dim];

    // array hosting the local indices for a given permutation
    size_t *idxes_ptr = new size_t[op_dim * batch_dim];


    for (size_t step = 0; step < steps; step++) {

#pragma omp parallel default(none) \
            shared(step, steps, psi_ptr, idxes_ptr, op_ptr, \
                   batch_dim, op_dim, r0, start, end, dims_ptr, \
                   newod_ptr, prods_ptr, nspins, comm) \
            private(s,b,lidx)
{
#pragma omp for
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {
            s = (step *  batch_dim + batch_step) * op_dim;
            b = batch_step * op_dim;

            //  compute permutation of index and move it to local indeces
            for (size_t j = 0; j < op_dim; j++)
                idxes_ptr[b+j] = g2l( permute_idx(s+j, dims_ptr, newod_ptr, prods_ptr, nspins ),
                                       start, end);
        }


#pragma omp for
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;

            // dot product
            for (size_t i = 0; i < op_dim; i++) {
                // reset r0[i]
                r0[i+b] = {0.0,0.0};
                for (size_t j = 0; j < op_dim; j++) {
                    lidx = idxes_ptr[b+j];

                    // check is inside our elements
                    if ( lidx == SIZE_MAX ) continue;

                    r0[i+b] += op_ptr[i*op_dim + j] * psi_ptr[lidx];
                }
            }
        }
#pragma omp master
{
        mp::sum(r0, op_dim * batch_dim, comm); // no error checking here, maybe not wise.

} // end omp master
#pragma omp barrier


        // update psi
{
#pragma omp for
        for (size_t batch_step = 0; batch_step < batch_dim; batch_step++) {

            b = batch_step * op_dim;
            for (size_t i = 0; i < op_dim; i++) {
                lidx = idxes_ptr[b+i];

                // check is inside our elements
                if ( lidx == SIZE_MAX ) continue;

                psi_ptr[lidx] = r0[b+i];
            }
        }
} // omp for
} //omp
    }
    delete [] r0;

    // delete index stuff
    delete [] idxes_ptr; delete [] prods_ptr; delete [] newod_ptr;
}


template <typename T>
double measure_mpi(py::array_t<std::complex<T>, py::array::c_style> op,
                   py::array_t<std::complex<T>, py::array::c_style> psi,
                   py::object py_comm) {


    MPI_Comm comm = *get_mpi_comm(py_comm);

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

    m.def("measure_mpi", &measure_mpi<double>, "Compute observable appearing as the last operator in the Hilbert space");
    m.def("measure_mpi", &measure_mpi<float>, "Compute observable appearing as the last operator in the Hilbert space");
    m.def("evolve_mpi", &evolve_mpi<double>, "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");
    m.def("evolve_mpi", &evolve_mpi<float>, "Evolves wavefunction in the Celio approach. The order in the Hilbert space must be ... x Nucleus x Muon");

}
