import numpy as np
import imageio
from math import sin, cos, acos, radians
import math
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, log


def get_video_config(video_name):
    file_in_1 = '/home/ml/Data/Video_All/' + video_name + '.mp4'
    # # get the paramters
    # # get the paramters
    video = imageio.get_reader(file_in_1, 'ffmpeg')

    FRAMERATE= round(video._meta['fps'])
    FRAMESCOUNT = video._meta['nframes']
    Frame_size  = video._meta['source_size']
    IMAGEWIDTH = round(Frame_size[0])
    IMAGEHEIGHT = round(Frame_size[1])
    Second_total = FRAMESCOUNT / FRAMERATE

    return FRAMERATE,  FRAMESCOUNT,  IMAGEWIDTH, IMAGEHEIGHT
