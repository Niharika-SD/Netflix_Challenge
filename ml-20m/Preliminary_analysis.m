clear all
close all

%% Experiments on medium dataset
load data_matrix_sm1.mat
load data_matrix_sm2.mat

x_1 = data_matrix_sm1;
x_2 = data_matrix_sm2;

x_med = horzcat(x_1,x_2);
y_med_act = vertcat(ones(size(x_1,2),1),2*ones(size(x_1,2),1));

k = find(x_med==0);
miss = ones(size(x_med));
miss(k) = 0;

[A_med,A1_med,rmse1_med,rmse2_med] = matrix_completion(x_med,miss,'step1',10);
[y_labels_1_med,y_labels_2_med] = clustering(A_med,A1_med,2);

%% Experiments on small datasets
load data_matrix_sm1.mat
load data_matrix_sm2.mat

x_small1 = x_1;
k = find(x_small1==0);
miss = ones(size(x_small1));
miss(k) = 0;

[A_small1,A1_small1,rmse1_small1,rmse2_small1] = matrix_completion(x_small1,miss,'step1',10);


x_small2 = x_2;
k = find(x_small2==0);
miss = ones(size(x_small2));
miss(k) = 0;

[A_small2,A1_small2,rmse1_small2,rmse2_small2] = matrix_completion(x_small2,miss,'step1',10);

%% Experiment on the entire dataset
load large_data_matrix.mat

x_large2 = large_data_matrix;
k = find(x_large2==0);
miss = ones(size(x_large2));
miss(k) = 0;

[A_small2,A1_small2,rmse1_small2,rmse2_small2] = matrix_completion(x_large2,miss,'step1',10);


%% create pseudo large dataset to study sequential matrix completion and clustering performance
load ldata_matrix_sm1.mat
load ldata_matrix_sm2.mat
load ldata_matrix_sm3.mat
load ldata_matrix_sm4.mat

x_1 = ldata_matrix_sm1;
x_2 = ldata_matrix_sm2;
x_3 = ldata_matrix_sm3;
x_4 = ldata_matrix_sm4;

x = horzcat(x_1,x_2,x_3,x_4);
y_act = vertcat(ones(size(x_1,2),1),2*ones(size(x_1,2),1),3 *ones(size(x_3,2),1),4 *ones(size(x_4,2),1));
% x = large_data_matrix;
k = find(x==0);
miss = ones(size(x));
miss(k) = 0;

[A,A1,~,~] = matrix_completion(x,miss,'step2',1);
[y_labels_1,y_labels_2] = clustering(A,A1,4);

A_step2 = [];

for j = 1:4
   
    x_cat = horzcat(x',y_labels_1) ;
    miss_cat = horzcat(miss',y_labels_1) ;
    
    miss_subs = x_cat(miss_cat(:,end)==j,1:end-1);
    x_subs = (1-miss_subs).*(x_cat(x_cat(:,end)==j,1:end-1));
    
    x_subs = x_subs';
    miss_subs =miss_subs';
    mu_user = sum(x_subs,2)./sum((1-miss_subs),2);
    mu_movie = sum(x_subs,1)./sum((1-miss_subs),1);

   
    x_aug =  x_subs+ (1-miss_subs).*(mu_user* ones(1,size(x_subs,2))*0.452 +0.548*ones(size(x_subs,1),1)*mu_movie);
    [m,n] = size(x_subs);
    tau = 5*sqrt(m*n);
    A_new = lrmc(x_aug,100*tau,miss_subs,0.2);
    A_step2 = horzcat(A_step2,A_new);
       
end



