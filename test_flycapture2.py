import flycapture2 as fc2
import numpy as np

def test():
    print fc2.get_library_version()
    c = fc2.Context()
    print c.get_num_of_cameras()
    c.connect(*c.get_camera_from_index(0))
    print c.get_camera_info()
    #c.set_video_mode_and_frame_rate(fc2.FC2_VIDEOMODE_1280x960Y8,
    #        fc2.FC2_FRAMERATE_7_5)
    #b = np.empty((6, 1280*960), dtype=np.uint8)
    #c.set_user_buffers(b)
    c.start_capture()
    im = fc2.Image()
    print [c.retrieve_buffer(im) for i in range(8)]
    print np.array(im)
    c.stop_capture()
    c.disconnect()

if __name__ == "__main__":
    test()
