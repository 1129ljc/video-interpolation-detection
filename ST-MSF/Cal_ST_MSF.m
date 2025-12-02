function [result,label] = Cal_ST_MSF(videos,status,T,num)
%% ST-MSF feature extractor from a videofile name(e.g. 'C:\test\akiyo_qcif.yuv');
%% input
% videos: video files in (rows,cols,nframes)format;
% status: the compression status;
% T: a threshold  for Markov feature extractor;
% num: iteration numbers;
%% output
% result: the set of ST-MSF;
% label: label interpolated frames as '1', static frame as '-1' after SCD and SSD;

length = size(videos,3);
label = zeros(1,length);
result = [];
if strcmp(status,'uncompressed')
    % for uncompressed videos 
    temp = videos(:,:,1) - videos(:,:,2);
    feaC = Markov(temp,T);
    result = [result;feaC];
    for kk = 2:length-1
        cur_frame = videos(:,:,kk);
        pre_frame = videos(:,:,kk-1);
        post_frame = videos(:,:,kk+1);
        if num == 1
            if SCD(cur_frame,pre_frame)
                if SSD(cur_frame,post_frame,post_frame)
                    label(1,kk) = 1;
                end
                continue;
            end
            if SSD(cur_frame,pre_frame,post_frame)
                label(1,kk) = -1;
                continue;
            end
        end
        temp = (2*cur_frame-pre_frame-post_frame)/2;
        feaC = Markov(temp,T);
        result = [result;feaC];
        
    end
    temp = videos(:,:,length) - videos(:,:,length-1);
    feaC = Markov(temp,T);
    result = [result;feaC];
else
    % for compressed videos
    quality_factor = 100;  %设置为100来捕捉JPEG的块内和块间变化；
    jpegfile1 = 'jpeg1.jpg';
    jpegfile2 = 'jpeg2.jpg';
    jpegfile3 = 'jpeg3.jpg';
    imwrite(videos(:,:,1),jpegfile1,'quality',quality_factor);
    imwrite(videos(:,:,2),jpegfile2,'quality',quality_factor);
    [D1,~] = deal(DCTPlane(jpegfile1),double(imread(jpegfile1)));
    [D2,~] = deal(DCTPlane(jpegfile2),double(imread(jpegfile2)));
    D = D1 - D2;
    feaC = Markov(D,T);
    result = [result;feaC];
    for kk = 2:length-1
        cur_frame = videos(:,:,kk);
        pre_frame = videos(:,:,kk-1);
        post_frame = videos(:,:,kk+1);
        if num == 1
            if SCD(cur_frame,pre_frame)
                if SSD(cur_frame,post_frame,post_frame)
                    label(1,kk) = 1;
                end
                continue;
            end
            if SSD(cur_frame,pre_frame,post_frame)
                label(1,kk) = -1;
                continue;
            end
        end
        imwrite(pre_frame,jpegfile1,'quality',quality_factor);
        imwrite(cur_frame,jpegfile2,'quality',quality_factor);
        imwrite(post_frame,jpegfile3,'quality',quality_factor);
        [D1,~] = deal(DCTPlane(jpegfile1),double(imread(jpegfile1)));
        [D2,~] = deal(DCTPlane(jpegfile2),double(imread(jpegfile2)));
        [D3,~] = deal(DCTPlane(jpegfile3),double(imread(jpegfile3)));
        D = (2*D2-D1-D3)/2;
        feaC = Markov(D,T);
        result = [result;feaC];
    end
    imwrite(videos(:,:,length),jpegfile1,'quality',quality_factor);
    imwrite(videos(:,:,length-1),jpegfile2,'quality',quality_factor);
    [D1,~] = deal(DCTPlane(jpegfile1),double(imread(jpegfile1)));
    [D2,~] = deal(DCTPlane(jpegfile2),double(imread(jpegfile2)));
    D = D1 - D2;
    feaC = Markov(D,T);
    result = [result;feaC];
end



