function [] = sampling(W,b,c,K)
% testing on the training data
% [train,val,test] = loadData1();
% pos = floor(rand([1 100])*3000);
% x = train.images(:,pos);

x = randn([784,100]);
% x = val.images(:,62:567);
%adding ones
x = [x; ones([1 size(x,2)])];

%binarize
% x = double(x>0.5);

% load('epochs10.mat')
W_c = [W ;c'];
W_b = [W b];
% K = 1000;
[x_t,h_t,x_tild,h_tild,reco] = CD(W_b,W_c,x,K,[]);

im_weights = visualize(reco');
figure;
imshow(im_weights')
end
