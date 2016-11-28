function [z2,a1,d1,mask] = forwardPropogation(X1,X1_b,b1,W1,b2,W2,N,output_dimension,p)
    
W1_b = [W1 b1];

%first layer
z1 = W1_b * X1_b';
a1 = sigmoid(z1);

if (p~=0)
    mask = (rand(size(a1)) < p) / (1-p);
else
    mask = ones(size(a1));
end
    d1 = a1 .* mask;
    d1_b = [d1 ; ones([1 N])];%[1 3000]
    W2_b = [W2 b2];
    z2 = W2_b * d1_b;

%%check for numerical stability
mat_max = repmat(max(z2),[output_dimension 1]);

end