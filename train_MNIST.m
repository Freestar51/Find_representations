disp('Data Loading...');
load('C:\Users\freestar\Documents\MATLAB\Library\DeepLearnToolbox-master\data\mnist_train.mat');    % train_X, train_labels
load('C:\Users\freestar\Documents\MATLAB\Library\DeepLearnToolbox-master\data\mnist_test.mat');     % test_X, test_labels
num_classes = 10;
train_x = train_X;
test_x = test_X;
train_y = [];
for idx = 1:num_classes
    train_y(:,idx) = train_labels==idx;
end
test_y = test_labels;
clear train_X test_X train_labels test_labels;
disp('Data Loaded, Start pretraining');

rng(0);
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   10;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   0.1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
disp('Done pretraining, Save pretrained data');
save('./dbn_pretrained.mat','dbn');
%unfold dbn to nn
disp('Start training...');
nn = dbnunfoldtonn(dbn, num_classes);
nn.activation_function = 'sigm';


%train nn
% opts.numepochs =  10;
% opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
disp('Done training, Save trained data');
save('./nn_trained.mat','nn');

pred = nnpredict(nn, test_x);
rand_idx = randperm(size(test_x,1));
for idx=1:100
    i = rand_idx(idx);
    x = test_x(i,:);
    imshow(reshape(x,[28 28])');
    title(sprintf('%d', pred(i)-1));
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    
    pause();
end
