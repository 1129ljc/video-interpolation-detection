function result = SCD(cur_frame,pre_frame)
%% implementation of paper in JDT, VOL. 8, NO. 3, MARCH 2012
% paper: Scene Change Detection Using Multiple Histograms for Motion-Compensated Frame Rate Up-Conversion
%% input
% cur_frame: current frame;
% pre_frame: previous frame;
%% output
% result: the result of scene change, 1: changed; 0: unchanged;
if size(cur_frame,3) == 3
    cur_frame = rgb2gray(cur_frame);
end
if size(pre_frame,3) == 3
    pre_frame = rgb2gray(pre_frame);
end
m = 32;
n = 32;
theta = 3;
lapta = 0.5;
fun = @graythresh;
Clevel = blkproc(cur_frame,[m n],fun);  
Plevel = blkproc(pre_frame,[m n],fun);
Clevel = fix(Clevel*256);
Plevel = fix(Plevel*256);
temp = abs(Clevel - Plevel);
ori_judge = temp > theta;
fun = @mean;
temp = blkproc(ori_judge,[3,3],fun);
sc = find(temp>lapta);
if isempty(sc)
    result = 0;
else
    result = 1;
end
end




  