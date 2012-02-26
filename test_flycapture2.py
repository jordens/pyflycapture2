import flycapture2 as fc2
import numpy as np

def test():
    print fc2.get_library_version()
    c = fc2.Context()
    print c.get_num_of_cameras()
    c.connect(*c.get_camera_from_index(0))
    print c.get_camera_info()
    c.set_video_mode_and_frame_rate(fc2.VIDEOMODE_1280x960Y8,
            fc2.FRAMERATE_7_5)
    m, f = c.get_video_mode_and_frame_rate()
    print m, f
    print c.get_video_mode_and_frame_rate_info(m, f)
    print c.get_property_info(fc2.FRAME_RATE)
    p = c.get_property(fc2.FRAME_RATE)
    print p
    c.set_property(**p)
    c.start_capture()
    im = fc2.Image()
    print [np.array(c.retrieve_buffer(im)).sum() for i in range(4)]
    a = np.array(im)
    print a.shape, a.base
    c.stop_capture()
    c.disconnect()

if __name__ == "__main__":
    test()
