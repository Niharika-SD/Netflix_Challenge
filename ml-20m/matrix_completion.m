function [A,A1,rmse1,rmse2] = matrix_completion(x,miss,strcase,cv_state)
    [m,n] = size(x);
    tau = 5*sqrt(m*n);
    
    k1 = find((1-miss)~=0);
    i_end = cv_state;
    for i = 1:i_end
        if strcmp(strcase,'step1')
            p = k1(randperm(size(k1,1)));
            k_1_train = p(1:ceil(0.9*size(k1,1)),1);
            k_1_test = p(ceil(0.9*size(k1,1))+1:end,1);
   
            miss(k_1_test)=0;
            x_train = x;
            x_train(k_1_test) =0;
        else
            x_train=x;
        end    
        pert = 1-miss;
    
        Gaus = rand(size(x));
        x1 = x_train + 5*Gaus.* pert;
        A = lrmc(x1,tau,miss,0.0002);
        if strcmp(strcase,'step1')
            rmse1(i) = sqrt(sum((x(k_1_test)-A(k_1_test)).*(x(k_1_test)-A(k_1_test)))/(size(k_1_test,1)));
        else 
            rmse1 = 0;
        end
        
        if strcmp(strcase,'step1')
           mu_user = sum(x_train,2)./sum(pert,2);
           mu_movie = sum(x_train,1)./sum(pert,1);

   
           x_aug =  x_train+ pert.*(mu_user* ones(1,size(x_train,2))*0.548 +0.452*ones(size(x_train,1),1)*mu_movie);
           A1 = lrmc(x_aug,tau,miss,0.02);
           rmse2(i) = sqrt(sum((x(k_1_test)-A1(k_1_test)).*(x(k_1_test)-A1(k_1_test))))/(size(k_1_test,1));
        else 
            A1= A;
            rmse2 = 0;
        end
    end
end