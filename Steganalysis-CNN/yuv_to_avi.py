'''
@Time    : 2021/5/20 10:41
@Author  : ljc
@FileName: yuv_to_avi.py
@Software: PyCharm
'''

import os

y4m_dir = 'F:/video_interpolation/video_dataset/AVI_FPS30/HMHE/CIF/'
yuv_dir = 'F:/video_interpolation/video_dataset/AVI_FPS30/HMHE/CIF/'
names = os.listdir(y4m_dir)

for idx in range(len(names)):
    video_name = names[idx]
    video_name_file = os.path.join(y4m_dir, video_name)
    video_name_prefix = video_name.split('.')[0] + '.avi'
    video_name_file_dst = os.path.join(yuv_dir, video_name_prefix)
    if not os.path.exists(video_name_file_dst):
        cmd = 'ffmpeg -s 352*288 -pix_fmt yuv420p -r 30 -i {yuv_file} -vcodec copy {avi_file}'.format(
            yuv_file=video_name_file,
            avi_file=video_name_file_dst
        )
        print(cmd)
        os.system(cmd)
