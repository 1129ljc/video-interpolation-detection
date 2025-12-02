function [efps,result] = MLDM(videofile,format,length,T,TRN_orig,TRN_method,fps)
%% input
% videofile: the file of a video, including the whole path;
% format: the format of yuv file, eg. 'qcif' or 'cif';
% length: the extracted frame length;
% T: the threshold of Markov;
% TRN_orig: ST-MSFs from original video trainning set;
% TRN_method: ST-MSFs from FRUCed video trainning set;
% fps: the fps of the candidate video;
%% output 
% efps: estimated original frame rate;
% result: ST-MSFs from choosed forgery frames
[videos,status] = videoread(videofile,format,length);
num = 1;
[Video_ST_MSF,label] = Cal_ST_MSF(videos,status,T,num);
maxiterationnum = 5;
Nup = inf;
Ts = size(videos,3)/fps;
result = [];
while (maxiterationnum >0 || Nup > Ts)
    iteration = 10;
    length = size(Video_ST_MSF,1);
    decision = zeros(1,length);
    for seed = 1:iteration
        [trained_ensemble,~] = ensemble_training(TRN_orig,TRN_method);
        test_results = ensemble_testing(Video_ST_MSF,trained_ensemble);
        decision = decision + test_results.predictions;
    end
    decision = decision/iteration;
    decision(decision>=0.5) = 1;
    decision(decision < 0.5) = 0;
    location = find(decision == 1);
    first = location(1,1); 
    last = location(1,end);
    Nup = 0;
    i = 1;
    upset = [];
    intvideo(:,:,i) = videos(:,:,first);
    k = first+1;
    while k < last
        i = i + 1;
        cur = decision(1,k);
        pre = decision(1,k-1);
        post = decision(1,k+1);
        if ((cur == 1) && (pre == 1) && (post ==1))
            Nup = Nup + 1;
            intvideo(:,:,i) = videos(:,:,k+1);
            upset = [upset;Video_ST_MSF(k,:)];
            k = k + 2;
        else
            intvideo(:,:,i) = videos(:,:,k);
            k = k + 1;
        end
    end
    intvideo(:,:,i+1) = videos(:,:,last);
    result = [result;upset];
    num = num + 1;
    [Video_ST_MSF,~] = Cal_ST_MSF(intvideo,status,T,num);
    maxiterationnum = maxiterationnum - 1;
    Ts = ceil((last-first)/fps);
end
num_up_frames = size(result,1) + sum(label==1);
efps = (size(videos,3)-num_up_frames)/(size(videos,3)/fps);
end
            
            
        
    



    

