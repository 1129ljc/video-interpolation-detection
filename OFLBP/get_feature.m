function feature = get_feature(videoFrames)
%OFLBP 提取视频optical flow LBP特征
%   videoFrames 视频灰度序列
%   feature     提取的特征

[H,W,len] = size(videoFrames);
feature_OFLBP = zeros(H,W,len-1);
for i=1:len-1
    % disp(['The ',num2str(i),'-th video frame optical flow LBP feature ...']);
    pre_frame = videoFrames(:,:,i);
    post_frame = videoFrames(:,:,i+1);
    optical_flow = opticalFlow(pre_frame,post_frame);
    Mag = optical_flow.Magnitude;
    LBP = LBP_extract(Mag);
    feature_OFLBP(:,:,i) = LBP;
end

feature = zeros(256,len);
for i=1:len
    % disp(['The ',num2str(i),'-th video frame optical flow LBP hist feature ...']);
    if i==1
        pre_frame = videoFrames(:,:,i);
        post_frame = videoFrames(:,:,i+1);
        weight = pre_frame - post_frame;
        LBP_feature = feature_OFLBP(:,:,i);
        OFLBP_hist = OFLBP_histogram(weight, LBP_feature);
    elseif i==len
        pre_frame = videoFrames(:,:,i-1);
        post_frame = videoFrames(:,:,i);
        weight = post_frame - pre_frame;
        LBP_feature = -feature_OFLBP(:,:,i-1);
        OFLBP_hist = OFLBP_histogram(weight, LBP_feature);
    else
        pre_frame = videoFrames(:,:,i-1);
        cur_frame = videoFrames(:,:,i);
        post_frame = videoFrames(:,:,i+1);
        weight = 2*cur_frame - post_frame - pre_frame;
        LBP_feature = feature_OFLBP(:,:,i);
        OFLBP_hist = OFLBP_histogram(weight, LBP_feature);
    end
    feature(:,i) = OFLBP_hist;
end
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
