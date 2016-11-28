% function [] = testing_5c(W,b,c,K)

folder_path = ['C:\Users\uzair\Desktop\DBM_Images'];

% testing on the training data
% [train,val,test] = loadData1();
% pos = floor(rand([1 100])*3000);
% x = train.images(:,pos);
no_chains = 100;
hidden = 100;
% no_chains = hidden;
x = randn([784,no_chains]);
h2 = randn([hidden,no_chains]);

%adding ones


%binarize
% x = double(x>0.0005);


gibbs = 1000;
[~,~,~,~,~,~,v_out,~] = PD(x,h2,W1,W2,b1,b2,c,no_chains,gibbs);
         

rec = visualize(v_out');
figure;
imshow(rec')

% filename = [num2str(plot_count) 'REhidden1=' num2str(hidden_layer1) 'hidden2=' num2str(hidden_layer2)];
filename = ['reconstruction' num2str(hidden)];
saveas(gca, fullfile(folder_path, filename), 'jpeg');