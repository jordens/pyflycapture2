from cwrap.config import Config, File

if __name__ == '__main__':
    config = Config('gccxml', files=[
	File('FlyCapture2Defs_C.h'),
	File('FlyCapture2_C.h'),
	])
    config.generate()
