function OFLBP_hist = OFLBP(pre_frame_path,cur_frame_path,post_frame_path,feature_path)

pre_frame = imread(pre_frame_path);
cur_frame = imread(cur_frame_path);
post_frame = imread(post_frame_path);
pre_frame = rgb2gray(pre_frame);
cur_frame = rgb2gray(cur_frame);
post_frame = rgb2gray(post_frame);

[H,W] = size(pre_frame);

% if H>288 && W>352
%     xmin = randi(H-288);
%     ymin = randi(W-352);
%     width = 352;
%     height = 288;
%     pre_frame = pre_frame(xmin:xmin+height-1, ymin:ymin+width-1);
%     cur_frame = cur_frame(xmin:xmin+height-1, ymin:ymin+width-1);
%     post_frame = post_frame(xmin:xmin+height-1, ymin:ymin+width-1);
% elseif H<288 && W>352
%     x_num = 1;
%     while x_num*H<288
%         x_num = x_num+1;
%     end
%     xmin = 0;
%     ymin = 0;
%     width = 352;
%     height = 288;
%     pre_frame_unit = pre_frame;
%     cur_frame_unit = cur_frame;
%     post_frame_unit = post_frame;
%     for i=1:x_num-1
%         pre_frame = [pre_frame;pre_frame_unit];
%         cur_frame = [cur_frame;cur_frame_unit];
%         post_frame = [post_frame;post_frame_unit];
%     end
%     pre_frame = pre_frame(xmin:xmin+height, ymin:ymin+width);
%     cur_frame = cur_frame(xmin:xmin+height, ymin:ymin+width);
%     post_frame = post_frame(xmin:xmin+height, ymin:ymin+width);
% elseif H>288 && W<352
%     y_num = 1;
%     while y_num*W<352
%         y_num = y_num+1;
%     end
%     xmin = 0;
%     ymin = 0;
%     width = 352;
%     height = 288;
%     pre_frame_unit = pre_frame;
%     cur_frame_unit = cur_frame;
%     post_frame_unit = post_frame;
%     for i=1:y_num-1
%         pre_frame = [pre_frame,pre_frame_unit];
%         cur_frame = [cur_frame,cur_frame_unit];
%         post_frame = [post_frame,post_frame_unit];
%     end
%     pre_frame = pre_frame(xmin:xmin+height, ymin:ymin+width);
%     cur_frame = cur_frame(xmin:xmin+height, ymin:ymin+width);
%     post_frame = post_frame(xmin:xmin+height, ymin:ymin+width);
% elseif H<288 && W<352  
%     x_num = 1;
%     while x_num*H<288
%         x_num = x_num+1;
%     end
%     y_num = 1;
%     while y_num*W<352
%         y_num = y_num+1;
%     end
%     xmin = 1;
%     ymin = 1;
%     width = 352;
%     height = 288;
%     pre_frame_unit = pre_frame;
%     cur_frame_unit = cur_frame;
%     post_frame_unit = post_frame;
%     for i=1:y_num-1
%         pre_frame = [pre_frame;pre_frame_unit];
%         cur_frame = [cur_frame;cur_frame_unit];
%         post_frame = [post_frame;post_frame_unit];
%     end
%     pre_frame_unit = pre_frame;
%     cur_frame_unit = cur_frame;
%     post_frame_unit = post_frame;
%     for i=1:x_num-1
%         pre_frame = [pre_frame,pre_frame_unit];
%         cur_frame = [cur_frame,cur_frame_unit];
%         post_frame = [post_frame,post_frame_unit];
%     end
%     pre_frame = pre_frame(xmin:height, ymin:width);
%     cur_frame = cur_frame(xmin:height, ymin:width);
%     post_frame = post_frame(xmin:height, ymin:width);
% end

weight = 2*cur_frame - post_frame - pre_frame;
pre_frame = double(pre_frame);
cur_frame = double(cur_frame);
post_frame = double(post_frame);
optical_flow = opticalFlow(pre_frame,cur_frame);
Mag = optical_flow.Magnitude;
LBP = LBP_extract(Mag);
OFLBP_hist = OFLBP_histogram(weight, LBP);
% save(feature_path,'OFLBP_hist');
end

function LBP = LBP_extract(image)
%提取单图像的LBP特征
[H,W] = size(image);
LBP = zeros(H-2,W-2);
for i=2:1:H-1
    for j=2:1:W-1
        neighbor = zeros(8);
        neighbor(1)=image(i-1,j-1);
        neighbor(2)=image(i-1,j);
        neighbor(3)=image(i-1,j+1);
        neighbor(4)=image(i,j-1);
        neighbor(5)=image(i,j+1);
        neighbor(6)=image(i+1,j-1);
        neighbor(7)=image(i+1,j);
        neighbor(8)=image(i+1,j+1);
        center=image(i,j);
        temp=uint8(0);
        for k=1:8
             temp =temp+ (neighbor(k) >= center)* 2^(k-1);
        end
        LBP(i-1,j-1) = temp;
    end
end
LBP = padarray(LBP,[1,1],'symmetric','both');
end


function weighted_feature = OFLBP_histogram(weight, LBP_feature)
%提取加权统计直方图特征
weighted_feature = zeros(256,1);
[H,W] = size(LBP_feature);
for i=1:H
    for j=1:W
        index = uint8(LBP_feature(i,j));
        if index ~=0
            weighted_feature(index,1) = weighted_feature(index,1) + weight(i,j);
        end
    end
end
end
