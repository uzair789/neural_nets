function [loss,accuracy,error] = test(data,data_labels,W1,b1,W2,b2,lambda,output_dimension)

N = size(data_labels,1);

data_b = [data ones([N 1])];%[3000 1]

[z2,~,~,~] = forwardPropogation(data,data_b,b1,W1,b2,W2,N,output_dimension,0);

[softmax_loss,class_probs,~] = softmax(output_dimension,z2,N,data_labels,lambda,W1,W2);
 
loss = softmax_loss;

%%compute the accuracy
[accuracy,error] = computeAccuracySoftmax(class_probs,data_labels);

end