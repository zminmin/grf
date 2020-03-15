
data_num = 10;
rng(10);
W = rand(data_num, 6);
Y = sum(W,2) + normrnd(0, 0.5, [data_num,1]);

M = [Y, W];
%writematrix(M, "test_custom_1.csv");
for i = 1:7
    fprintf('%.4f, ',M(:,i));
    if i == 7
        fprintf('\n');
    end
end

weight = [0.0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.0, 0.1];

Xx = [ones(data_num,1), W];
beta = (Xx' * Xx) \ Xx' * Y;
weight_beta = (Xx' * diag(weight) * Xx) \ Xx' * diag(weight) * Y;
epsilon = [0,0,0,1,1,1];
%beta = [1,2,3,4,5,6,7]';
barY = mean(Y);
barW = mean(W, 1); 
Ap = zeros(6, 6);
for i = 1:data_num
    Ap = Ap + (W(i, :) - barW)' * (W(i, :) - barW);
end
Yhat = W * beta(2:7,1);
Ap = Ap / data_num;
cons = epsilon / (Ap);
rho = zeros(data_num, 1);
for i = 1:data_num
    rho(i, 1) = cons * (W(i, :) - barW)' * (Y(i) - barY - (W(i, :) - barW) * beta(2:7,1));
    fprintf('%.4f, ',rho(i));
    if i == data_num
       fprintf('\n');
    end
end



    