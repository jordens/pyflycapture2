from _FlyCapture2Defs_C cimport *
from _FlyCapture2_C cimport *

class ApiError(BaseException):
    pass

cdef check_error(fc2Error e):
    if e != FC2_ERROR_OK:
        raise ApiError(e, fc2ErrorToDescription(e))

cdef class Context:
    cdef fc2Context ctx

    def create_context(self):
        check_error(fc2CreateContext(&self.ctx))

    def __cinit__(self):
        self.create_context()

    def destroy(self):
        check_error(fc2DestroyContext(self.ctx))

    def __dealloc__(self):
        self.destroy()

    def get_num_of_cameras(self):
        cdef unsigned int n
        check_error(fc2GetNumOfCameras(self.ctx, &n))
        return n

    def get_num_of_devices(self):
        cdef unsigned int n
        check_error(fc2GetNumOfDevices(self.ctx, &n))
        return n

    def get_camera_from_index(self, unsigned int index):
        cdef fc2PGRGuid g
        cdef Guid guid = Guid()
        check_error(fc2GetCameraFromIndex(self.ctx, index, &g))
        guid.guid = g
        return guid

    def connect(self, Guid g not None):
        check_error(fc2Connect(self.ctx, &g.guid))

    def disconnect(self):
        check_error(fc2Disconnect(self.ctx))


cdef class Guid:
    cdef fc2PGRGuid guid
