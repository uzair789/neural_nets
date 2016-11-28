function [loss,accuracy,error] = test(data,data_labels,W1,b1,W2,b2,lambda,output_dimension)
N = size(data_labels,1);

data_b = [data ones([N 1])];%[3000 1]

% [class_probs,~] = forwardPropogation(data,data_b,ones1,W1,ones2,W2,N);
[z2,~,~,~] = forwardPropogation(data,data_b,b1,W1,b2,W2,N,output_dimension,0);

[softmax_loss,class_probs,~] = softmax(output_dimension,z2,N,data_labels,lambda,W1,W2);


% for i = 1:N;
%     correct_class(i) = class_probs(data_labels(i)+1,i); %% check this if getting errors
%     
%     %computing the gradient of softmax here to avoid another for loop
% %     class_probs2(train_labels(i)+1,i) = class_probs2(train_labels(i)+1,i) - 1;
% end
% 
% log_f = -log(correct_class);
% data_loss =(N_inv) * sum(log_f);
% reg_loss = 0.5 * lambda * sum(sum(W1.*W1)) + 0.5 * lambda * sum(sum(W2.*W2));
% 
loss = softmax_loss;

%%compute the accuracy
[accuracy,error] = computeAccuracySoftmax(class_probs,data_labels);

end