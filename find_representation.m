% Visulization of Deep Belief Network
% Understanding Representations Learned in Deep Architectures, D. Erhan
%
% Sourced from DeepLearnToolbox:
%                dbn.m
%                dbn_mnist_D.mat
%
% Written by Giyoung Jeon
% Probabilistic Artificial Intelligence Lab at UNIST
% v1.0 June, 11th, 2015


addpath(genpath('./'));
load('nn_trained.mat');

rho=10;
figure;
% mult = 0:10:990;
% u_idx = floor(mod(rand(1,100)*100,10));
% u_idx = u_idx + mult;
x_ = cell(1,100);
parfor idx = 1:100
    disp(sprintf('optimizing unit %d\n',idx));
    x_{idx} = grad_ascent(nn,rho,idx);
end

