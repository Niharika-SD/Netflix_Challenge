function [L,E]=rpca_admm(X,tau,method)
%implements robust PCA solution by alternative directions of maximisation
%algorithm
thresh = 0.02;
max_iter =1000;

if (strcmp(method,'gross_errors'))
    E = zeros(size(X));
    L= E;
    lambda_mat = E;
    lambda =0.001;
    tau_inv = 1/tau;
    lam_by_tau = lambda/tau;
    
   for i = 1:max_iter
    
    E_prev = E;
    L_prev = L;
    %updates
    fprintf('Iteration number: %d || error  = %f \n',i, norm(X -L- E,'fro'));
    [U,S,V] =svd(X-E+tau_inv.*lambda_mat);
    S =sign(S).*(max(abs(S)-tau_inv,0));
    L = U*S*V' ;
    temp =(X-L+tau_inv*lambda_mat);
    E = sign(temp).*(max(abs(temp)-lam_by_tau,0));
    lambda_mat = lambda_mat + tau*(X-L-E);
    
    %check for convergence
    if(norm((E_prev-E),'fro')<=thresh&& norm((L_prev-L),'fro')<=thresh)
    %if(norm(X -L- E,'fro')<=thresh) 
        break;
    end
   end
   
elseif((strcmp(method,'outliers')))
     E = zeros(size(X));
     E_prev = E ;
     L = E;
     L_prev = E;
     lambda = tau;
     delta = 10e-05;
     nu = 0.9;
     mu_naught = 0.99* norm(X,'fro');
     mu = mu_naught;
     t_prev = 1;
     t =1;
     mu_bar = delta*mu;
     for i = 1: max_iter
        fprintf('Iteration number: %d || error  = %f \n',i, norm(X -L- E,'fro'));
        Y_L = L +((t_prev-1)/t)*(L-L_prev);
        Y_E = E + ((t_prev-1)/t)*(E-E_prev);
        G_L = Y_L -0.5*(Y_L+Y_E-X);
        G_E = Y_E -0.5*(Y_L+Y_E-X);
     
     [U,S,V] =svd(G_L);
     S =sign(S).*(max(abs(S)-mu/2,0));
     L_prev =L;
     L = U*S*V';
     E_prev = E;
     for j =1:size(X,2)
         if(norm(G_E(:,j),2)<lambda*mu/2)
             E(:,j) = zeros(size(X,1),1);
         else
             E(:,j) = G_E(:,j)-(lambda*mu/2)*G_E(:,j)/norm(G_E(:,j),2);
         end
     end
     t_prev =t ;
     t = (1+ sqrt(4*t^2+1))/2 ;
     mu = max(nu*mu,mu_bar);     
     if(norm((E_prev-E),'fro')<=thresh&& norm((L_prev-L),'fro')<=thresh)
        break;
     end
     end
    
end

end
