#ifndef UNDI_THREADS_HPP
#define UNDI_THREADS_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

// Maximum number of OpenMP threads the extension will use (set by OMP_NUM_THREADS).
// It is 1 if the extension was built without OpenMP.
inline int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#endif
