'''
@Time    : 2021/5/19 18:55
@Author  : ljc
@FileName: y4m_to_yuv.py
@Software: PyCharm
'''

import os

y4m_dir = 'F:/video_interpolation/video_dataset2/original1/1080p/'
yuv_dir = 'F:/video_interpolation/video_dataset2/original1/1080p_yuv/'
names = os.listdir(y4m_dir)

for idx in range(len(names)):
    video_name = names[idx]
    video_name_file = os.path.join(y4m_dir, video_name)
    video_name_prefix = video_name.split('.')[0] + '.yuv'
    video_name_file_dst = os.path.join(yuv_dir, video_name_prefix)
    if not os.path.exists(video_name_file_dst):
        cmd = 'ffmpeg -i {input} -vsync 0 {output} -y'.format(
            input=video_name_file,
            output=video_name_file_dst
        )
        print(cmd)
        os.system(cmd)
