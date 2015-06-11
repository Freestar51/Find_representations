% Visulization of Deep Belief Network
% Understanding Representations Learned in Deep Architectures, D. Erhan
%
% Sourced from DeepLearnToolbox:
%                DBN
%                NN
%
% Written by Giyoung Jeon
% Probabilistic Artificial Intelligence Lab at UNIST
% v1.0 June, 11th, 2015


addpath(genpath('./'));
load('nn_trained.mat');

rho=10;
x_ = cell(1,100);
if(~exist('representations.mat'))
    parfor idx = 1:100
        x_{idx} = grad_ascent(nn,rho,idx);
    end
    save('representations.mat','x_');
else
    load('representations.mat');
end

load('data/mnist_test.m');
x_ = cell2mat(x_);
mult = 0:10:990;
u_idx = floor(mod(rand(1,100)*100,10));
u_idx = u_idx + mult;
for idx=1:100
    ridx = u_idx(idx);
    nn = nnff(test_X(ridx,:),zeros(size(X,1), nn.size(end)));
    repr = nn.a{end-1}'*x_;
    subplot(1,2,1);
    imshow(reshape(test_X(ridx,:),[28 28])');
    subplot(1,2,2);
    imshow(reshape(repr, [28 28])');
    pause();
end
