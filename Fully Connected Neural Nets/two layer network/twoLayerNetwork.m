clear all;
tic;

%%initializing the collectors
collector_lr = [];
collector_m = [];
collector_lambda = [];
collector_dropout1 = [];
collector_dropout2 = [];
collector_epochs = [];
collector_hidden1 = [];
collector_hidden2 = [];
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

learning_rates = [0.5];%[0.01,0.1,0.2,0.5];
momentum = [0.9];%][0,0.5,0.9];
reg = [0];%
P1 = [0.5]; % p = 0 will shut drop out
P2 = [0.5]; % p = 0 will shut drop out
NUM_epochs = [300]; 
hidden_layer1_sizes = [300];
hidden_layer2_sizes = [500];%[20,100,200,500];

for i = 1 : length(learning_rates)
    for j = 1 : length(momentum)
        for k = 1 : length(reg)
            for ii = 1 : length(P1)
                for mm = 1 : length(P2)
                    for jj = 1 : length(NUM_epochs)
                        for kk = 1 : length(hidden_layer1_sizes)
                            for ll = 1 : length(hidden_layer2_sizes)
                
%%hyper-parameters
lr = learning_rates(i);
m = momentum(j); %momentum
lambda = reg(k);
lr_init = lr;

v_W1 = 0; v_W2 = 0; v_W3 = 0; v_b3 = 0; v_b1 = 0; v_b2 = 0;
num_epochs = NUM_epochs(jj);
p1 = P1(ii); %for dropout
p2 = P2(mm);

%%Annealing
anneal_switch = 1;  %1 is on and 0 is off
anneal_epoch = 500;
anneal_rate = 0.5;

%%network architecture 784 -> 100 -> dropout1 -> 100 -> dropout2 -> 10
input_dimension = 784;
hidden_layer1 = hidden_layer1_sizes(kk);
hidden_layer2 = hidden_layer2_sizes(ll);
output_dimension = 10;

%initialize the weights and the bias

W1 = sqrt(6)/sqrt(hidden_layer1+input_dimension) * randn([hidden_layer1 input_dimension]);%zeros(hidden_layer,input_dimension);%([100,784]);
b1 = zeros([hidden_layer1 1]);%([100 1]);

W2 = sqrt(6)/sqrt(hidden_layer1 + hidden_layer2) * randn([hidden_layer2 hidden_layer1]);%([10 100]);
b2 = zeros([hidden_layer2 1]);%([100 1]);

W3 = sqrt(6)/sqrt(hidden_layer2 + output_dimension) * randn([output_dimension hidden_layer2]);
b3 = zeros([output_dimension 1]);

X1 = train_data;
X1_b = [train_data ones([N 1])];%[3000 1]

total_train_loss = [];total_train_accuracy = [];
total_val_loss = [];total_val_accuracy = [];d1=0;mask=[];

for epoch = 1:num_epochs
%%%%%%%%%Forward Propogation%%%%%%%%%
[z3,a1,a2,d1,d2,mask1,mask2] = forwardPropogation(X1,X1_b,b1,W1,b2,W2,N,output_dimension,W3,b3,p1,p2);

[softmax_loss,class_probs,class_probs2] = softmax(output_dimension,z3,N,train_labels,lambda,W1,W2,W3);

%%
%%%%%%%%Back propogation%%%%%
%%Computing the Gradients 
d_loss_layer = class_probs2/N;

db3 = sum(d_loss_layer,2) ; 

dW3 = d_loss_layer * d2';

dd2 = W3' * d_loss_layer;

da2 = mask2 .* dd2;

dz2 = (a2) .* (ones(size(a2)) - a2) .* da2; 

db2 = sum(dz2,2);

dW2 = dz2 * d1';

dd1 = W2' * dz2;

da1 = mask1 .* dd1;

dz1 = (a1) .* (ones(size(a1)) - a1) .* da1; 

db1 = sum(dz1,2);

dW1 = dz1 * X1;

%%adding the regularizer derivatives
dW3 = dW3 + lambda * W3;
dW2 = dW2 + lambda * W2;
dW1 = dW1 + lambda * W1;

%updating with momentum
v_W3 = m * v_W3 - lr * dW3;
W3 = W3 + v_W3;

v_W2 = m * v_W2 - lr * dW2;
W2 = W2 + v_W2;

v_W1 = m * v_W1 - lr * dW1;
W1 = W1 + v_W1;

v_b3 = m * v_b3 - lr * db3;
b3 = b3 + v_b3;

v_b2 = m * v_b2 - lr * db2;
b2 = b2 + v_b2;

v_b1 = m * v_b1 - lr * db1;
b1 = b1 + v_b1;

%testing on the training and validation data

[train_loss,train_accuracy,train_error] = test(train_data,train_labels,W1,b1,W2,b2,lambda,output_dimension,W3,b3);
total_train_accuracy(epoch) = train_accuracy;
total_train_error(epoch) = train_error;
total_train_loss(epoch) = train_loss;

[val_loss,val_accuracy,val_error] = test(val_data,val_labels,W1,b1,W2,b2,lambda,output_dimension,W3,b3);
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
[test_loss,test_accuracy,test_error] = test(test_data,test_labels,W1,b1,W2,b2,lambda,output_dimension,W3,b3)

%collector arrays
collector_lr = [collector_lr lr];
collector_m = [collector_m m];
collector_lambda = [collector_lambda lambda];
collector_dropout1 = [collector_dropout1 p1];
collector_dropout2 = [collector_dropout2 p2];
collector_epochs = [collector_epochs num_epochs];
collector_hidden1 = [collector_hidden1 hidden_layer1];
collector_hidden2 = [collector_hidden2 hidden_layer2];
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
network = sprintf('%d --> %d --> %d --> %d',input_dimension,hidden_layer1,hidden_layer2,output_dimension);
hyperparams = sprintf('lr = %.3f | momentum = %.3f | lambda = %.3f | Dropout1 = %.2f | Dropout2 = %.2f | anneal switch = %d',lr_init,m,lambda,p1,p2,anneal_switch);
train_vals = sprintf('Train loss = %.4f | Train accuracy = %.2f | Train error = %.2f',train_loss,train_accuracy,train_error);
test_vals = sprintf('Test loss = %.4f | Test accuracy = %.2f | Test error = %.2f',test_loss,test_accuracy,test_error);
val_vals = sprintf('Val loss = %.4f | Val accuracy = %.2f | Val error = %.2f',val_loss,val_accuracy,val_error);
anneal_vals = sprintf('annealing epoch = %d | annealing rate = %.2f',anneal_epoch,anneal_rate);

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
units = (ylim(2)-ylim(1))/10;
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


figure(4);
plot(xaxis,total_train_error,'-b',xaxis,total_val_error,'--r');
legend('training error','validation error');
title('Training error and Validation error');
xlabel('epochs');
ylabel('error');
ylim = get(gca,'Ylim');
units = (ylim(2)-ylim(1))/10;
xpos = 30;
ypos = max(total_train_error)-units;
text(xpos,ypos,network);
text(xpos,ypos-units,hyperparams);
text(xpos,ypos-2*units,test_vals)
text(xpos,ypos-3*units,train_vals);
text(xpos,(ypos-4*units),val_vals);

                    
end%lr
end %p1
end %p2
end %momentum
end%reg
end % NUM_epochs
end%hidden_layers1
end%hidden_layers2

[max_val,max_pos] = max(collector_test_accuracy);
header = sprintf('Best hyperparameters for expCount = %d, Test accuracy = %.2f , Test loss = %.4f, Test error = %.4f, Train accuracy = %.2f, Train loss = %.4f , Train error = %.4f, val accuracy = %.2f, val loss = %.4f, val error = %.4f',collector_count(max_pos), collector_test_accuracy(max_pos),collector_test_loss(max_pos),collector_test_error(max_pos),collector_train_accuracy(max_pos),collector_train_loss(max_pos),collector_train_error(max_pos), collector_val_accuracy(max_pos),collector_val_loss(max_pos),collector_val_error(max_pos));
disp('best hyperparameters')

bestparams = sprintf('lr = %.3f | momentum = %.4f | lambda = %.4f | Dropout1 = %.2f | Dropout2 = %.2f | numEpochs = %d | hiddenLayer1Size = %d | hiddenLayer2Size = %d ',collector_lr(max_pos),collector_m(max_pos),collector_lambda(max_pos),collector_dropout1(max_pos),collector_dropout2(max_pos),collector_epochs(max_pos),collector_hidden1(max_pos),collector_hidden2(max_pos));
disp(header);
disp(bestparams);

toc;


