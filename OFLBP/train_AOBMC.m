clc;
clear all;

CIF_mat_dir = "F:\video_interpolation\video_dataset2\MP4_FPS_30_PNG_OFBLP\QCIF\";
names = dir(fullfile(CIF_mat_dir));
FileNames = {names.name}; 
video_names = FileNames(3:end);

real_mat = [];
fake_mat = [];

for video_index=1:size(video_names, 2)
    video = video_names(video_index);
    feature_name_path = strcat(CIF_mat_dir,video,"\");
    disp(feature_name_path);
    names = dir(fullfile(feature_name_path));
    FileNames = {names.name}; 
    pics = FileNames(3:end);
    disp(size(pics,2));
    for i=2:size(pics, 2)-1
        mat_file = strcat(feature_name_path,num2str(i),".mat");
        mat = load(mat_file);
        feature = mat.OFLBP_hist;
        feature = transpose(feature);
        if mod(i,2)==0
            fake_mat = [fake_mat;feature];
        elseif mod(i,2)==1
            real_mat = [real_mat;feature];
        end
        
    end
end

[real_num,~] = size(real_mat);
[fake_num,~] = size(fake_mat);

if real_num<fake_num
    feature_num = real_num;
else
    feature_num = fake_num;
end

if mod(feature_num, 2) == 1
    feature_num = feature_num - 1;
end
real_mat = real_mat(1:feature_num,:);
fake_mat = fake_mat(1:feature_num,:);

disp(size(real_mat));
disp(size(fake_mat));

RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));

random_permutation_real = randperm(feature_num);
% training_set = random_permutation_real(1:round(feature_num/2));
training_set = random_permutation_real(1:(feature_num/2));
testing_set = random_permutation_real(round(feature_num/2)+1:end);

% Prepare training features
training_for_real = real_mat(training_set,:);
training_for_fake = fake_mat(training_set,:);
% training_for_real = real_mat;
% training_for_fake = fake_mat;

% Prepare testing features
testing_for_real = real_mat(testing_set,:);
testing_for_fake = fake_mat(testing_set,:);
% testing_for_real = real_mat;
% testing_for_fake = fake_mat;

disp(size(training_for_real));
disp(size(training_for_fake));

[trained_ensemble,results] = ensemble_training(training_for_real,training_for_fake);

save('trained.mat','trained_ensemble');
figure(1);
clf;plot(results.search.d_sub,results.search.OOB,'.b');hold on;
plot(results.optimal_d_sub,results.optimal_OOB,'or','MarkerSize',8);
xlabel('Subspace dimensionality');ylabel('OOB error');
legend({'all attempted dimensions',sprintf('optimal dimension %i',results.optimal_d_sub)});
title('Search for the optimal subspace dimensionality');

figure(2);
clf;plot(results.OOB_progress,'.-b');
xlabel('Number of base learners');ylabel('OOB error')
title('Progress of the OOB error estimate');

test_results_real = ensemble_testing(testing_for_real,trained_ensemble);
test_results_fake = ensemble_testing(testing_for_fake,trained_ensemble);

false_alarms = sum(test_results_real.predictions~=-1);
missed_detections = sum(test_results_fake.predictions~=+1);
num_right = sum(test_results_real.predictions==-1)+sum(test_results_fake.predictions==+1);
num_testing_samples = size(testing_for_real,1)+size(testing_for_fake,1);
testing_error = (false_alarms + missed_detections)/num_testing_samples;
testing_accuracy = num_right/num_testing_samples;
fprintf('Testing error: %.4f\n',testing_error);
fprintf('Testing accuracy: %.4f\n',testing_accuracy);

figure(3);clf;
[hc,x] = hist(test_results_real.votes,50);
bar(x,hc,'b');hold on;
[hs,x] = hist(test_results_fake.votes,50);
bar(x,hs,'r');hold on;
legend({'real','fake'});
xlabel('majority voting');
ylabel('histogram');

labels = [-ones(size(testing_for_real,1),1);ones(size(testing_for_fake,1),1)];
votes  = [test_results_real.votes;test_results_fake.votes];
[X,Y,T,auc] = perfcurve(labels,votes,1);
figure(4);clf;plot(X,Y);hold on;plot([0 1],[0 1],':k');
xlabel('False positive rate'); ylabel('True positive rate');title('ROC');
legend(sprintf('AUC = %.4f',auc));

