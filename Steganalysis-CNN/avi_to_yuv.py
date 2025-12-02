'''
@Time    : 2021/9/8 15:59
@Author  : ljc
@FileName: avi_to_yuv.py
@Software: PyCharm
'''

import os

video_dir = 'F:/video_interpolation/video_dataset/AVI_FPS15/CIF/'
video_dst_dir = 'F:/video_interpolation/video_dataset/AVI_FPS30/HMHE/CIF/'

names = os.listdir(video_dir)

for i in range(len(names)):
    name = names[i].split('.')[0]
    video = os.path.join(video_dir, name + '.avi')
    video_dst = os.path.join(video_dst_dir, name + '.yuv')
    cmd = 'ffmpeg -loglevel quiet -i {} {}'.format(video, video_dst)
    print(cmd)
    os.system(cmd)
