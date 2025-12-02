function result = SSD(cur_frame,pre_frame,post_frame)
%% implementation of Static Scene Detection 
%% input
% cur_frame: current frame;
% pre_frame: previous frame;
% post_frame: post frame;
%% output
% result: the result of static scene, 1: static frame; 0: non-static frame;
if size(cur_frame,3) == 3
    cur_frame = rgb2gray(cur_frame);
end
if size(pre_frame,3) == 3
    pre_frame = rgb2gray(pre_frame);
end
if size(post_frame,3) == 3
    post_frame = rgb2gray(post_frame);
end
temp = uint8(abs((2*cur_frame-pre_frame-post_frame)/2)); % fix 取最小整数； round：四舍五入
data = sum(temp(:)==0);
[im_rows,im_cols] = size(cur_frame);
num = im_rows * im_cols;
if (data/num >= 0.995)
    result = 1;
else
    result = 0;
end
end