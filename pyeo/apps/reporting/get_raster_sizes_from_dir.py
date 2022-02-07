"""
get raster file sizes from all jp2 and tiff files in a directory
can be useful for checking correct download of Sentinel-2 SAFE directories
"""

import pyeo.raster_manipulation
import pyeo.filesystem_utilities
from pyeo.filesystem_utilities import get_filenames
from pyeo.raster_manipulation import get_stats_from_raster_file

import argparse
import os
from osgeo import gdal
from tempfile import TemporaryDirectory

gdal.UseExceptions()

def reports(path, logfile):
    log = pyeo.filesystem_utilities.init_log(logfile)
    # get all image paths of tiff and jpeg2000 files
    log.info(path)
    image_paths = [ os.path.join(p,f) for p,d,fs in os.walk(path) for f in fs \
                    if os.path.isfile(os.path.join(p, f)) \
                    and ( f.endswith(".tif") or f.endswith(".jp2") \
                    or f.endswith(".jpg") or f.endswith(".jpeg") ) ]
    if len(image_paths) == 0:
         raise FileNotFoundError("No tiff or jpeg images found in {}.".format(path))
    for index, image in enumerate(image_paths):
        size = os.path.getsize(image)
        if size < 10*1024:
            log.info("{}: {} Bytes  {}".format(index+1, str(size), image))
        else:
            if size < 10*1024*1024:
                log.info("{}: {} KB  {}".format(index+1, str(round(size/1024)), image))
            else:
                if size < 10*1024*1024*1024:
                    log.info("{}: {} MB  {}".format(index+1, str(round(size/1024/1024)), image))
                else:
                    log.info("{}: {} GB  {}".format(index+1, str(round(size/1024/1024/1024)), image))
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick file size reporting for tiff and jpeg2000 files.')
    parser.add_argument(dest='path', action='store', default=r'~',
                        help="A path to a directory.")
    parser.add_argument(dest='logfile', action='store', default=r'~/raster_stats_log.txt',
                        help="A path to the log file.")
    args = parser.parse_args()

    reports(**vars(args))
