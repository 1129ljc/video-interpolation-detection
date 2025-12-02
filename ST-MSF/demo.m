%% Note:
% in fold <FRUC_codes>, there are five FRUC methods: AOBMC, DSME, EBME, 
% MCMP, and PFRUC; for the dualME and MHME, Please request the executed
% codes from the authors of the papers [1],[2];
% [1] S. J.Kang, S. J. Yoo, and Y. H.Kim, ¡°Dual motion estimation for frame
% rate up-conversion,¡± IEEE Trans. Circuits Syst. Video Technol., vol. 20,
% no. 12, pp. 1909-1914, 2010.
% [2] H. B. Liu, R. Q. Xin, D. B. Zhao, S. W. Ma, and W. Gao, ¡°Multiple
% hypotheses bayesian frame rate up-conversion by adaptive fusion of
% motion-compensated interpolations,¡± IEEE Trans. Circuits Syst. Video
% Technol., vol. 22, no. 8, pp. 1188-1198, 2012.

% in fold <utilities>, there have the details implementation function which
% will be called by the execution of the FRUC method.
% in fold <ECOC_classifier>, the implementation of Error-Correcting Output 
% Code (ECOC) strategy based on Ensemble classifier is here;
% the Ensemble classifier can be download from http://dde.binghamton.edu/download/ensemble, 
% For the use of this classifier, please refer to the tutorial in the downloaded zip file;

%% extract the ST-MSF feature after SCD and SSD for a candidate video;
clc;
clear all;
addpath('..\VideoTools\');
addpath('..\VideoRead\');
num = 1;
T = 3;

% for uncompressed video: akiyo_qcif.yuv
% videofile = 'akiyo_qcif.yuv';
% [videos,status] = videoread(videofile,format,length);
% [Video_ST_MSF1,label1] = Cal_ST_MSF(videos,status,T,num);

% for compressed video: VideoSeq_1.avi
% videofile = 'VideoSeq_1.avi';
videofile = 'E:\frame_interpolation_matlab\video_dataset\AVI_FPS15\CIF\akiyo_cif.avi';
videoFrames = AVIReadGray(videofile,[1,1]);
disp(size(videoFrames));
[Video_ST_MSF2,label2] = Cal_ST_MSF(videoFrames,'uncompressed',3,0);