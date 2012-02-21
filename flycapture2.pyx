from _FlyCapture2_C_merged cimport *

cdef class FlyCapture2:
    cdef fc2Context *_ctx
    def __cinit__(self):
        self.check_error(fc2CreateContext(self._ctx))

    def get_num_of_cameras(self):
        cdef unsigned int n
        self.check_error(fc2GetNumOfCameras(self._ctx, &n))
        return n
