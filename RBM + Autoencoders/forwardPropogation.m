function [z2,a1,d1,mask] = forwardPropogation(X1,X1_b,b1,W1,b2,W2,N,output_dimension,p)

    
W1_b = [W1 b1];

%first layer
z1 = W1_b * X1_b';
a1 = sigmoid(z1);

if (p~=0)
%     disp(['in p = ' num2str(p)]);
    %%implement dropout here
    mask = (rand(size(a1)) < p) / (1-p);
    %scaling should be done by p only but i do with  1-p here because i
    %assume that dropout is off when p = 0. 
    %ANother way to do it would be to remove the if else and shut dropout
    %when p =1. when p = 1, mask will be identity.
    %p means that percent of neurons are active
    %initialy i think mistook p for number of dead neurons..and hence i was
    %dividing with 1-p which i think now is wrong.
%     mask = (mask < p) / p;
else
    mask = ones(size(a1));
%     d1 = a1;
end
    d1 = a1 .* mask;
    d1_b = [d1 ; ones([1 N])];%[1 3000]
    W2_b = [W2 b2];
    z2 = W2_b * d1_b;
% else
% %     disp(['in p = ' num2str(p)]);
%     W2_b = [W2 b2];
%     a1_b = [a1 ; ones([1 N])];%[1 3000]
%     z2 = W2_b * a1_b;
%     
% end

%%check for numerical stability
mat_max = repmat(max(z2),[output_dimension 1]);

end