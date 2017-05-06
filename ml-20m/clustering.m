function [y_labels_1,y_labels_2] = clustering(A,A1,n_clust)

    [y_labels_1,C1] = kmeans(A',n_clust);
    

    figure
    A_norm = (A'-(ones(size(A,2),1)*mean(A')))./(ones(size(A,2),1)*std(A'));
    [silh1,h1] = silhouette(A_norm,y_labels_1,'Euclidean');
    h1 = gca;
    h1.Children.EdgeColor = [.8 .8 1];
    xlabel 'Silhouette Value'
    ylabel 'Cluster'
    
    if ~strcmp(A1,'junk') 
        [y_labels_2,C2] = kmeans(A1',n_clust);
        figure
        A1_norm = (A1'-(ones(size(A1,2),1)*mean(A1')))./(ones(size(A1,2),1)*std(A1'));
        [silh2,h2] = silhouette(A1'./(ones(size(A1,2),1)*max(A1')),y_labels_2,'Euclidean');
        h2 = gca;
        h2.Children.EdgeColor = [.8 .8 1];
        xlabel 'Silhouette Value'
        ylabel 'Cluster'
    else 
        y_labels_2 = 0;
    end
    
end