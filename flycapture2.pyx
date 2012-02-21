from _FlyCapture2_C_merged cimport *

class ApiError(BaseException):
    pass

cdef class Context:
    cdef fc2Context _ctx

    def check_error(self, fc2Error e):
        if e != FC2_ERROR_OK:
            raise ApiError(e, fc2ErrorToDescription(e))

    def __cinit__(self):
        self.check_error(fc2CreateContext(&self._ctx))

    def get_num_of_cameras(self):
        cdef unsigned int n
        self.check_error(fc2GetNumOfCameras(&self._ctx, &n))
        return n

    def get_num_of_devices(self):
        cdef unsigned int n
        self.check_error(fc2GetNumOfDevices(&self._ctx, &n))
        return n
