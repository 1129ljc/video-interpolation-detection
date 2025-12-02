clc;
clear all;
addpath('F:\video_interpolation\video_tools\VideoRead\');

mat_file = 'trained.mat';
mat = load(mat_file);
trained_ensemble = mat.trained_ensemble;

video = 'F:\video_interpolation\video_dataset\AVI_FPS30\AOBMC\CIF\akiyo_cif.avi';
videoFrames = AVIReadGray(video,[1,1]);
feature = OFLBP(videoFrames);
feature = transpose(feature);
result = ensemble_testing(feature,trained_ensemble);
save(save_name,'result');



