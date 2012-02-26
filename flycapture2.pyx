from _FlyCapture2_C cimport *

import numpy as np
cimport numpy as np

from cpython cimport PyObject, Py_INCREF

np.import_array()

class ApiError(BaseException):
    pass

cdef raise_error(fc2Error e):
    if e != FC2_ERROR_OK:
        raise ApiError(e, fc2ErrorToDescription(e))

def get_library_version():
    cdef fc2Version v
    fc2GetLibraryVersion(&v)
    return {"major": v.major, "minor": v.minor,
            "type": v.type, "build": v.build}

cdef class Context:
    cdef fc2Context ctx

    def __cinit__(self):
        cdef fc2Error r
        with nogil:
            r = fc2CreateContext(&self.ctx)
        raise_error(r)

    def __dealloc__(self):
        cdef fc2Error r
        with nogil:
            r = fc2DestroyContext(self.ctx)
        raise_error(r)

    def get_num_of_cameras(self):
        cdef unsigned int n
        cdef fc2Error r
        with nogil:
            r = fc2GetNumOfCameras(self.ctx, &n)
        raise_error(r)
        return n

    def get_num_of_devices(self):
        cdef unsigned int n
        cdef fc2Error r
        with nogil:
            r = fc2GetNumOfDevices(self.ctx, &n)
        raise_error(r)
        return n

    def get_camera_from_index(self, unsigned int index):
        cdef fc2PGRGuid g
        cdef fc2Error r
        with nogil:
            r = fc2GetCameraFromIndex(self.ctx, index, &g)
        raise_error(r)
        return g.value[0], g.value[1], g.value[2], g.value[3]

    def get_camera_info(self):
        cdef fc2CameraInfo i
        cdef fc2Error r
        with nogil:
            r = fc2GetCameraInfo(self.ctx, &i)
        raise_error(r)
        ret = {"serial_number": i.serialNumber,
             "model_name": i.modelName,
             "vendor_name": i.vendorName,
             "sensor_info": i.sensorInfo,
             "sensor_resolution": i.sensorResolution,
             "firmware_version": i.firmwareVersion,
             "firmware_build_time": i.firmwareBuildTime,}
        return ret

    def connect(self, unsigned int a, unsigned int b,
            unsigned int c, unsigned int d):
        cdef fc2PGRGuid g
        cdef fc2Error r
        g.value[0], g.value[1], g.value[2], g.value[3] = a, b, c, d
        with nogil:
            r = fc2Connect(self.ctx, &g)
        raise_error(r)

    def disconnect(self):
        cdef fc2Error r
        with nogil:
            r = fc2Disconnect(self.ctx)
        raise_error(r)

    def set_video_mode_and_frame_rate(self, int mode, int framerate):
        cdef fc2Error r
        with nogil:
            r = fc2SetVideoModeAndFrameRate(self.ctx,
                <fc2VideoMode>mode, <fc2FrameRate>framerate)
        raise_error(r)

    def set_user_buffers(self,
            np.ndarray[np.uint8_t, ndim=2] buff not None):
        raise_error(fc2SetUserBuffers(self.ctx, <unsigned char *>buff.data,
            buff.shape[1], buff.shape[0]))

    def start_capture(self):
        cdef fc2Error r
        with nogil:
            r = fc2StartCapture(self.ctx)
        raise_error(r)

    def stop_capture(self):
        cdef fc2Error r
        with nogil:
            r = fc2StopCapture(self.ctx)
        raise_error(r)

    def retrieve_buffer(self, Image img=None):
        cdef fc2Error r
        if img is None:
            img = Image()
        with nogil:
            r = fc2RetrieveBuffer(self.ctx, &img.img)
        raise_error(r)
        return img

cdef class Image:
    cdef fc2Image img

    def __cinit__(self):
        cdef fc2Error r
        with nogil:
            r = fc2CreateImage(&self.img)
        raise_error(r)

    def __dealloc__(self):
        cdef fc2Error r
        with nogil:
            r = fc2DestroyImage(&self.img)
        raise_error(r)

    def __array__(self):
        cdef np.ndarray r
        cdef np.npy_intp shape[3]
        shape[0] = self.img.rows
        shape[1] = self.img.cols
        shape[2] = self.img.dataSize/self.img.rows/self.img.cols
        r = np.PyArray_SimpleNewFromData(3, shape, np.NPY_UINT8,
                self.img.pData)
        r.base = <PyObject *>self
        Py_INCREF(self)
        return r
