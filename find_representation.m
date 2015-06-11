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
    x=cell2mat(x_);
    save('representations.mat','x');
    clear x_;
else
    load('representations.mat');
end

load('data/mnist_test.mat');
pred = nnpredict(nn, test_X);
u_idx = find(pred~=test_labels);
% mult = 0:100:9900;
% u_idx = floor(mod(rand(1,100)*100,10));
% u_idx = u_idx + mult;
for idx=1:length(u_idx);
    ridx = u_idx(idx);
    nn = nnff(nn, test_X(ridx,:),zeros(1, nn.size(end)));
    repr = nn.a{end-1}*x./sum(x);
    repr = (repr-mean(repr))/std(repr);
    subplot(1,2,1);
    imshow(reshape(test_X(ridx,:),[28 28])');
    title(sprintf('original: %d',test_labels(ridx)-1));
    subplot(1,2,2);
    imshow(reshape(repr, [28 28])');
    title(sprintf('predicted: %d',pred(ridx)-1));
    pause();
end
