function A=lrmc(X,tau,W,thresh)
%Finds the low-rank approximation of a matrix X with incomplete entries as 
%specified in W using the low-rank matrix completion algorithm based on the 
%augmented Lagrangian method.

[m,n] = size(X);
Z = zeros(m,n);
maxIter =1000;

l_param = 1.9; %learning parameter for proximal gradient
A= Z;
figure;
title('Error convergence')
xlabel('no of iterations')
ylabel('Error')

for i = 1:maxIter
    fprintf('Iteration number: %d || error  = %f \n',i, norm(X-A ,'fro'));
    Z_dash = Z.*W;
    err(i) = norm(X-A ,'fro');
    [U,S,V] =svd(Z_dash);
    S =sign(S).*(max(abs(S)-tau,0));
    Z_prev =Z;
    A = U*S*V';
    Z = Z + l_param*(W.*X -W.*A);
    if (norm((Z-Z_prev),'fro')/(norm(Z,'fro'))<thresh)
        break;
    end
end
plot(1:i,err)

end