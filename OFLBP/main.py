import os
import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine


def get_video_quality(video_path):
    video_size = os.path.getsize(video_path)
    video_cap = cv2.VideoCapture(video_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    video_bts = ((video_size / 1024) * 8) / (frame_num / video_fps)
    video_ratio = frame_width * frame_height
    video_val = video_bts / video_ratio
    video_cap.release()
    return video_val


def opencv_trans(video_dir, video_save_dir):
    files = os.listdir(video_dir)
    for file in files:
        file_path = os.path.join(video_dir, file)
        print(file_path)
        image_save_path = os.path.join(video_save_dir, file.split('.')[0])
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)
        video_cap = cv2.VideoCapture(file_path)
        frame_num = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        for i in range(int(frame_num)):
            ref, frame = video_cap.read()
            image_save = os.path.join(image_save_path, str(i + 1) + '.png')
            cv2.imwrite(image_save, frame)
        video_cap.release()


def draw(judges, result_save):
    x = [i for i in range(len(judges))]
    y = judges
    barlist = plt.bar(x, y)
    for i in range(len(judges)):
        if judges[i] < 0:
            barlist[i].set_color('r')
    plt.title("Predict per frame")
    plt.xlabel("Frame3~Frame100")
    plt.ylabel("Ensemble classifier vote")
    plt.savefig(result_save)
    plt.close()


def test_videos(input_a, input_b, input_c):
    input_arg_task_id = input_a
    input_arg_file_path = input_b
    input_arg_ext = input_c

    input_arg_ext_json = json.loads(input_arg_ext)
    input_arg_ext_out_json_path = input_arg_ext_json['JSON_FILE_PATH']

    input_arg_ext_tmp_dir = input_arg_ext_json['TMP_DIR']
    input_arg_ext_tmp_dir = os.path.join(input_arg_ext_tmp_dir, 'ljc_docs')

    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_tmp_dir, 'interpolation_detection_by_OFLBP')
    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_out_tmp_path, str(input_arg_task_id))
    file_temp_dir = os.path.join(input_arg_ext_out_tmp_path, 'files')
    result_dir = os.path.join(input_arg_ext_out_tmp_path, 'result')

    if not os.path.exists(file_temp_dir):
        os.makedirs(file_temp_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    opencv_trans(input_arg_file_path, file_temp_dir)

    algorithm_message = '使用光流幅值的LBP特征作为权值，为帧间插值矩阵分配权重得到光流LBP帧差特征，使用基于Fisher线性判别的集成分类器进行分类。'
    print(algorithm_message)

    files = os.listdir(input_arg_file_path)
    result_json_content = {}
    eng = matlab.engine.start_matlab()

    for i in range(len(files)):
        video_name_org = os.path.join(input_arg_file_path, files[i])
        quality = get_video_quality(video_name_org)
        x = ''
        if quality < 0.01:
            x = '低质量模型'
        elif quality > 0.01 and quality < 0.02:
            x = '中质量模型'
        elif quality > 0.02:
            x = '高质量模型'
        video_detec_name = os.path.join(file_temp_dir, files[i].split('.')[0])
        video_cap = cv2.VideoCapture(video_name_org)
        frame_rate = int(video_cap.get(cv2.CAP_PROP_FPS))
        video_cap.release()
        result_save = os.path.join(result_dir, files[i].split('.')[0] + '.png')
        print('video_detec_name', video_name_org)
        print('result_save', result_save)
        ret = eng.test(video_detec_name)
        res = np.array(ret).astype(np.int)
        res = res.squeeze()
        result = np.sign(res)
        judges = res
        result = result
        if len(result) > 105:
            judges = res[3:103]
            result = result[3:103]

        draw(judges, result_save)

        if np.sum(result[0:-1:2]) > (len(result) // 4):
            conclusion = '该视频为伪造视频，原始帧率为' + str(frame_rate // 2)
            confidence = 1.0
        else:
            conclusion = '插帧周期特征不足，该视频为真实视频'
            confidence = 0.0
        image_feature = []
        image_feature_one = {'filepath': result_save,
                             'title': '集成分类器逐帧判别结果',
                             'comment': '该图展示集成分类器的判别结果，若观察到明显的周期性，则该视频为伪造视频。'}
        image_feature.append(image_feature_one)

        video_json = {'taskid': str(input_arg_task_id),
                      'conclusion': conclusion,
                      'message': algorithm_message,
                      'confidence': confidence,
                      'threshold': 0,
                      'features': image_feature,
                      'ext':{
                          '视频码率': quality,
                          '去适配模型信息': x
                      }}

        result_json_content[files[i]] = video_json
    eng.exit()
    json_path = input_arg_ext_out_json_path
    with open(json_path, 'w') as f:
        json.dump(result_json_content, f)
    f.close()


if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    input_3 = sys.argv[3]
    test_videos(input_a=input_1, input_b=input_2, input_c=input_3)
