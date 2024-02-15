struct Handler {
#if defined __MPI
  // to be removed soon
  MPI_Datatype *mpi_send_types = NULL;
  MPI_Datatype *mpi_recv_types = NULL;
  MPI_Comm comm;
  int rank=0;
  int size=1;
#endif
  size_t wfc_size; // local dimension of wfc
  size_t h_dim; // global dimension of Hilber space
  size_t nspins; // total number of spins
  size_t * dims_ptr = NULL; // total number of spins
  size_t * send_counts = NULL; // to be removed
  size_t * rotations = NULL;
  size_t * inv_rotations = NULL;
  bool initialized = false;
  bool mpi_initialized = false; // to be removed
  bool single_precision = false;
};
