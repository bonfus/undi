struct Handler {
  size_t wfc_size; // local dimension of wfc
  size_t h_dim; // global dimension of Hilber space
  size_t nspins; // total number of spins
  size_t * dims_ptr = NULL; // total number of spins
  size_t * rotations = NULL;
  size_t * inv_rotations = NULL;
  bool initialized = false;
};
