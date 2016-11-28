function [loss,class_probs,class_probs2] = softmax(output_dimension,input,N,labels,lambda,W1,W2)

expo = exp(input); 
sum_all_class = sum(expo,1);
dividing_matrix = repmat(sum_all_class,[output_dimension 1]);
class_probs = expo ./ dividing_matrix;

%making a copy for the gradient of softmax
class_probs2 = class_probs; 

%%Computing the loss function (Softmax)
for i = 1:N;
    correct_class(i) = class_probs(labels(i)+1,i); %% check this if getting errors
    
    %computing the gradient of softmax here to avoid another for loop
    class_probs2(labels(i)+1,i) = class_probs2(labels(i)+1,i) - 1;
end

log_f = -log(correct_class);
data_loss = (1/N) * sum(log_f);
reg_loss = 0.5 * lambda * sum(sum(W1.*W1)) + 1/2 * lambda * sum(sum(W2.*W2));
loss = data_loss + reg_loss;
end