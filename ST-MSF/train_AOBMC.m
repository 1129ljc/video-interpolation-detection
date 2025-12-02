clc;
clear all;

video_names = ["akiyo_cif.mat", "bowing_cif.mat", "bridge_close_cif.mat", "bus_cif.mat", ...
    "city_cif.mat", "coastguard_cif.mat", "container_cif.mat", "crew_cif.mat", ...
    "flower_cif.mat", "football_cif.mat", "hall_monitor_cif.mat", "harbour_cif.mat", ...
    "highway_cif_1.mat", "highway_cif_2.mat", "highway_cif_3.mat", "highway_cif_4.mat", ....
    "highway_cif_5.mat", "highway_cif_6.mat", "highway_cif_7.mat", "ice_cif.mat", ...
    "mobile_cif.mat", "mother_daughter_cif.mat", "news_cif.mat", "silent_cif.mat", ...
    "tempete_cif.mat", "waterfall_cif.mat"];

AOBMC_mat_dir = 'F:\video_interpolation\detection_methods\ST-MSF\feature_extract\AOBMC\CIF_30_no_compression\';
DSME_mat_dir = 'F:\video_interpolation\detection_methods\ST-MSF\feature_extract\DSME\CIF_30_no_compression\';
EBME_mat_dir = 'F:\video_interpolation\detection_methods\ST-MSF\feature_extract\EBME\CIF_30_no_compression\';
MCMP_mat_dir = 'F:\video_interpolation\detection_methods\ST-MSF\feature_extract\MCMP\CIF_30_no_compression\';
MFM_mat_dir = 'F:\video_interpolation\detection_methods\ST-MSF\feature_extract\MFM\CIF_30_no_compression\';
PFRUC_mat_dir = 'F:\video_interpolation\detection_methods\ST-MSF\feature_extract\PFRUC\CIF_30_no_compression\';

real_mat2 = [];
fake_mat2 = [];
for i=1:size(video_names,2)
    mat_file = strcat(AOBMC_mat_dir,video_names(1,i));
    mat = load(mat_file);
    feature = mat.Video_ST_MSF2;
    [num,~] = size(feature);
    real_data = feature(1:2:num,:);
    fake_data = feature(2:2:num,:);
    real_mat2 = [real_mat2;real_data];
    fake_mat2 = [fake_mat2;fake_data];
end

real_mat = real_mat2;
fake_mat = fake_mat2;

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
training_set = random_permutation_real(1:round(feature_num/2));
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
save('trained.mat','trained_ensemble')
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

