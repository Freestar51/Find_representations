function [ X_ ] = grad_ascent( nn, rho, n )
% Simplified Gradient Ascent for Visualize DBN
%
% Input Parameters:
%               nn:      learned NN with DBN of DeepLearnToolbox
%               rho:    constraint of norm(x) during the gradient ascent
%               n:      the index of visualizing unit
%
% Written by Giyoung Jeon
% Probabilistic Artificial Intelligence Lab at UNIST
% v1.3 June, 11th, 2015

    function y = tarfun(X, nn)
        nn_ = nnff(nn, X, zeros(size(X,1), nn.size(end)));
        y = nn_.a{end-1}(n);
    end

    maxiter=100;
    learn_rate = 1;
    delta = 14/1000;
    sample_n = 10;
    X_ = zeros(1,size(nn.a{1},2));
    for samp_i=1:sample_n
        X0 = rand(28);
        X0 = X0 / norm(X0)*rho;
        X0 = reshape(X0, 28*28,1)';
        for iter=1:maxiter
            grad = zeros(size(X0));
            for i=1:size(X0,2)
                X = X0;
                y1 = tarfun(X,nn);
                X(i) = X(i)+delta;
                y2 = tarfun(X,nn);
                grad(i) = (y2-y1)/(delta);
                X0= X0 + grad*learn_rate;
                X0=X0/norm(X0)*rho;
            end
            ex_grad = grad;
        end
        X_ = X_ + X0;
%         if(max(abs(ex_grad-grad))<0.00001)
%             break
%         end
    end
    X_=X_/sample_n;
end