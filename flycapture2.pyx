from _FlyCapture2_C cimport *

import numpy as np
cimport numpy as np

class ApiError(BaseException):
    pass

cdef check_error(fc2Error e):
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
        check_error(fc2CreateContext(&self.ctx))

    def __dealloc__(self):
        check_error(fc2DestroyContext(self.ctx))

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
        cdef unsigned int a, b, c, d
        check_error(fc2GetCameraFromIndex(self.ctx, index, &g))
        a = g.value[0]
        b = g.value[1]
        c = g.value[2]
        d = g.value[3]
        return a, b, c, d

    def get_camera_info(self):
        cdef fc2CameraInfo i
        check_error(fc2GetCameraInfo(self.ctx, &i))
        r = {"serial_number": i.serialNumber,
             "model_name": i.modelName,
             "vendor_name": i.vendorName,
             "sensor_info": i.sensorInfo,
             "sensor_resolution": i.sensorResolution,
             "firmware_version": i.firmwareVersion,
             "firmware_build_time": i.firmwareBuildTime,}
        return r

    def connect(self, unsigned int a, unsigned int b, unsigned int c,
            unsigned int d):
        cdef fc2PGRGuid g
        g.value[0] = a
        g.value[1] = b
        g.value[2] = c
        g.value[3] = d
        check_error(fc2Connect(self.ctx, &g))

    def disconnect(self):
        check_error(fc2Disconnect(self.ctx))

    def set_video_mode_and_frame_rate(self, int mode, int framerate):
        check_error(fc2SetVideoModeAndFrameRate(self.ctx,
            <fc2VideoMode>mode, <fc2FrameRate>framerate))

    def set_user_buffers(self,
            np.ndarray[np.uint8_t, ndim=2] buff not None):
        check_error(fc2SetUserBuffers(self.ctx, <unsigned char *>buff.data,
            buff.shape[1], buff.shape[0]))

    def start_capture(self):
        check_error(fc2StartCapture(self.ctx))

    def stop_capture(self):
        check_error(fc2StopCapture(self.ctx))

    def retrieve_buffer(self, Image img=None):
        if img is None:
            img = Image()
        check_error(fc2RetrieveBuffer(self.ctx, &img.img))
        return img

cdef class Image:
    cdef fc2Image img
    def __cinit__(self):
        check_error(fc2CreateImage(&self.img))
    def __dealloc__(self):
        check_error(fc2DestroyImage(&self.img))
