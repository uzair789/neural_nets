function [] = OneLayerNetwork(pretrain_type,W1,b1,p1,ep)
%pretrain_type is RBM, autoencoder or denoising autoencoder
%p1 is the plot count
%W1 abd b1 are the first layer weights and bias

tic;
if nargin <4
    plot_count = 0;
    disp('plot count is 0!');
else
    plot_count = p1;
end
%===== Change the folder path to your own location ====%
folder_path = 'C:\Users\uzair\Desktop\Images_1_layer';
mkdir(folder_path)

%%initializing the collectors
collector_lr = [];
collector_m = [];
collector_lambda = [];
collector_dropout = [];
collector_epochs = [];
collector_hidden = [];

collector_test_loss = [];
collector_test_accuracy = [];
collector_test_error = [];

collector_train_loss = [];
collector_train_accuracy = [];
collector_train_error = [];

collector_val_loss = [];
collector_val_accuracy = [];
collector_val_error = [];

collector_count = [];

%load the data

disp('Loading Data...');
 [train_data,train_labels,val_data,val_labels,test_data,test_labels] = loadData();
disp('Data Loaded!!');


%N is the number of data points
N = length(train_labels);
N_inv = 1/N;

%%%
%appending a row of ones on W1 and W2
X1 = train_data;
X1_b = [train_data ones([N 1])];%[3000 1]

learning_rates = [0.5];%[0.3,0.5,0.7];
momentum = [0.9];
reg = [0];%,0.001,0.004];%
P = [0]; % p = 0 will shut drop out and p = 0.5 will turn it on
NUM_epochs = [ep];
hidden_layer_sizes = [100];%[200,300,500];

for i = 1 : length(learning_rates)
    for j = 1 : length(momentum)
        for k = 1 : length(reg)
            for ii = 1 : length(P)
                for jj = 1 : length(NUM_epochs)
                    for kk = 1 : length(hidden_layer_sizes)
                
%%hyper-parameters
lr = learning_rates(i);
m = momentum(j); %momentum
lambda = reg(k);
lr_init = lr;

v_W1 = 0; v_W2 = 0; v_b1 = 0; v_b2 = 0;
num_epochs = NUM_epochs(jj);
p = P(ii); %for dropout

%%Annealing
anneal_switch = 0;  %1 is on and 0 is off
anneal_epoch = 500;
anneal_rate = 0.5;

%%network architecture 784 -> 100 -> dropout -> 10
input_dimension = 784;
hidden_layer = hidden_layer_sizes(kk);
output_dimension = 10;

%initialize the weights and the bias
% W1 = 0.01 * randn([hidden_layer,input_dimension]);%zeros(hidden_layer,input_dimension);%([100,784]);
if length(W1) ==1 && length(b1) == 0
    disp('Initializing random weights and bias!!')
    W1 = sqrt(6)/sqrt(hidden_layer+input_dimension) * randn([hidden_layer input_dimension]);%zeros(hidden_layer,input_dimension);%([100,784]);
    b1 = zeros([hidden_layer 1]);%([100 1]);
else
    disp('Using pretrained weights!!')
    W1=W1;b1=b1;
end
% W2 = 0.01 * randn([output_dimension hidden_layer]);%([10 100]);

W2 = sqrt(6)/sqrt(output_dimension + hidden_layer) * randn([output_dimension hidden_layer]);%([10 100]);
b2 = zeros([output_dimension 1]);%([10 1]);

total_train_loss = [];total_train_accuracy = [];total_train_error = [];
total_val_loss = [];total_val_accuracy = [];total_val_error = [];

for epoch = 1:num_epochs

    %%%%%%%%%Forward Propogation%%%%%%%%%

[z2,a1,d1,mask] = forwardPropogation(X1,X1_b,b1,W1,b2,W2,N,output_dimension,p);

[softmax_loss,class_probs,class_probs2] = softmax(output_dimension,z2,N,train_labels,lambda,W1,W2);

total_train_loss(epoch) = softmax_loss;


%%
%%%%%%%%Back propogation%%%%%
%%Computing the Gradients 
d_loss_layer = class_probs2/N;
db2 = sum(d_loss_layer,2) ; %db2 must be the same dimension as b2
dW2 = d_loss_layer * d1';%%%%remove if no dropout
dd1 = W2' * d_loss_layer;%%%%%remove if no dropout
da1 = mask .* dd1;%%%remove if no drop out
dz1 = (a1) .* (ones(size(a1)) - a1) .* da1; %%possible mistake....sigmoid derivative
db1 = sum(dz1,2);
dW1 = dz1 * X1;

%%adding the regularizer derivatives
dW2 = dW2 + lambda * W2;
dW1 = dW1 + lambda * W1;
%%
%%updating (vanilla update)the weights and the bias
% W2 = W2 - lr * dW2;
% b2 = b2 - lr * db2;
% W1 = W1 - lr * dW1;
% b1 = b1 - lr * db1;

%updating with momentum
v_W2 = m * v_W2 - lr * dW2;
W2 = W2 + v_W2;

v_W1 = m * v_W1 - lr * dW1;
W1 = W1 + v_W1;

v_b2 = m * v_b2 - lr * db2;
b2 = b2 + v_b2;

v_b1 = m * v_b1 - lr * db1;
b1 = b1 + v_b1;


%Computing the training and validation accuracies

[train_loss,train_accuracy,train_error] = test(train_data,train_labels,W1,b1,W2,b2,lambda,output_dimension);
total_train_accuracy(epoch) = train_accuracy;
total_train_error(epoch) = train_error;

[val_loss,val_accuracy,val_error] = test(val_data,val_labels,W1,b1,W2,b2,lambda,output_dimension);
total_val_loss(epoch) = val_loss;
total_val_accuracy(epoch) = val_accuracy;
total_val_error(epoch) = val_error;

if (anneal_switch == 1 && mod(epoch,anneal_epoch) == 0)
    disp('in annealing')
    lr = lr * anneal_rate;
end
disp(['epoch ' num2str(epoch) ' -- train loss ' num2str(total_train_loss(epoch)) ' -- val loss ' num2str(total_val_loss(epoch))]);
end

%%Testing the system on the test set
[test_loss,test_accuracy,test_error] = test(test_data,test_labels,W1,b1,W2,b2,lambda,output_dimension)

% %%create a collector matrix
% [lr m lambda p num_epochs hidden_layer train_loss train_accuracy]

%collector arrays
collector_lr = [collector_lr lr];
collector_m = [collector_m m];
collector_lambda = [collector_lambda lambda];
collector_dropout = [collector_dropout p];
collector_epochs = [collector_epochs num_epochs];
collector_hidden = [collector_hidden hidden_layer];
collector_test_loss = [collector_test_loss test_loss];
collector_test_accuracy = [collector_test_accuracy test_accuracy];
collector_test_error = [collector_test_error test_error];
collector_train_loss = [collector_train_loss train_loss];
collector_train_accuracy = [collector_train_accuracy train_accuracy];
collector_train_error = [collector_train_error train_error];
collector_val_loss = [collector_val_loss val_loss];
collector_val_accuracy = [collector_val_accuracy val_accuracy];
collector_val_error = [collector_val_error val_error];
collector_count = [collector_count plot_count];

%%for the display
network = sprintf('%d --> %d --> %d -- pretraining -- %s',input_dimension,hidden_layer,output_dimension, pretrain_type);
hyperparams = sprintf('lr = %.3f | momentum = %.3f | lambda = %.4f | Dropout = %.2f | anneal switch = %d',lr_init,m,lambda,p,anneal_switch);
train_vals = sprintf('Train loss = %.4f | Train accuracy = %.2f | Train error = %.2f',train_loss,train_accuracy,train_error);
test_vals = sprintf('Test loss = %.4f | Test accuracy = %.2f | Test error = %.2f',test_loss,test_accuracy,test_error);
val_vals = sprintf('Val loss = %.4f | Val accuracy = %.2f | Val error = %.2f',val_loss,val_accuracy,val_error);
anneal_vals = sprintf('annealing epoch = %d | annealing rate = %.3f',anneal_epoch,anneal_rate);

%visualize the weights
image_weights = visualize(W1);

figure(1);
imshow(image_weights);
title(network);
xlabel(hyperparams);
filename = [num2str(plot_count) ' weights for first layer'];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

xaxis = 1:num_epochs;

figure(2);
plot(xaxis,total_train_loss,'-b',xaxis,total_val_loss,'--r');
legend('training loss','validation loss');
title('Traning loss & Validation loss vs Epochs');
xlabel('epochs');
ylabel('loss values');
ylim = get(gca,'Ylim');
xlim = get(gca,'Xlim');
units = (ylim(2)-ylim(1))/10;
units_x = (xlim(2)-xlim(1))/10;
xpos = 50;
ypos = max(total_train_loss)-units;
text(xpos,ypos,network);
text(xpos,(ypos-units),hyperparams);
text(xpos,(ypos-2*units),test_vals);
text(xpos,(ypos-3*units),train_vals);
text(xpos,(ypos-4*units),val_vals);
if (anneal_switch == 1)
    text(xpos,ypos-5*units,anneal_vals);
end
filename = [num2str(plot_count) ' Training loss and Validation loss' ];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(3);
plot(xaxis,total_train_accuracy,'-b',xaxis,total_val_accuracy,'--r')
legend('training accuracy','validation accuracy')
title('Traning Accuracy & Validation Accuracy vs Epochs');
xlabel('epochs');
ylabel('accuracy values');
ylim = get(gca,'Ylim');
units = (ylim(2)-ylim(1))/10;
ypos = max(total_train_accuracy)-units;
text(xpos,ypos,network);
text(xpos,ypos-units,hyperparams);
text(xpos,ypos-2*units,test_vals)
text(xpos,ypos-3*units,train_vals);
text(xpos,(ypos-4*units),val_vals);
filename = [num2str(plot_count) ' Training accuracy and Validation accuracy'];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(4);
plot(xaxis,total_train_error,'-b',xaxis,total_val_error,'--r');
legend('training error','validation error');
title('Training error and Validation error');
xlabel('epochs');
ylabel('error');
ylim = get(gca,'Ylim');
units = (ylim(2)-ylim(1))/10;
ypos = max(total_train_error)-units;
text(xpos,ypos,network);
text(xpos,ypos-units,hyperparams);
text(xpos,ypos-2*units,test_vals)
text(xpos,ypos-3*units,train_vals);
text(xpos,(ypos-4*units),val_vals);
filename = [num2str(plot_count) ' Training error and Validation error'];
saveas(gca, fullfile(folder_path, filename), 'jpeg');
plot_count = plot_count + 1;
close all;

end%lr
end %p
end %momentum
end%reg
end % NUM_epochs
end%hidden_layers

[max_val,max_pos] = max(collector_test_accuracy);
header = sprintf('Best hyperparameters for expCount = %d, Test accuracy = %.2f , Test loss = %.4f, Train accuracy = %.2f, Train loss = %.4f, val accuracy = %.2f, val loss = %.4f',collector_count(max_pos), collector_test_accuracy(max_pos),collector_test_loss(max_pos),collector_train_accuracy(max_pos),collector_train_loss(max_pos),collector_val_accuracy(max_pos),collector_val_loss(max_pos));
disp('best hyperparameters');

bestparams = sprintf('lr = %.3f | momentum = %.4f | lambda = %.4f | Dropout = %.2f | numEpochs = %d | hiddenLayerSize = %d',collector_lr(max_pos),collector_m(max_pos),collector_lambda(max_pos),collector_dropout(max_pos),collector_epochs(max_pos),collector_hidden(max_pos));
disp(header)
disp(bestparams)
toc;

end



