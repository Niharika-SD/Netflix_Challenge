clear all
close all

load data_matrix_sm1.mat
load data_matrix_sm2.mat
load large_data_matrix.mat
x_1 = data_matrix_sm1;
x_2 = data_matrix_sm2;

x = horzcat(x_1,x_2);
y_act = vertcat(zeros(size(x_1,2),1),ones(size(x_1,2),1));
% x = large_data_matrix;


[y_labels_1,C1] = kmeans(A',2);
[y_labels_2,C2] = kmeans(A1',2);

figure
[silh1,h1] = silhouette(A'./(ones(size(A,2),1)*max(A')),y_labels_1,'Euclidean');
h1 = gca;
h1.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'

figure
[silh2,h2] = silhouette(A1'./(ones(size(A1,2),1)*max(A1')),y_labels_2,'Euclidean');
h2 = gca;
h2.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'

