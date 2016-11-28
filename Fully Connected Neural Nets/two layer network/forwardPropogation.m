function [z3,a1,a2,d1,d2,mask1,mask2] = forwardPropogation(X1,X1_b,b1,W1,b2,W2,N,output_dimension,W3,b3,p1,p2)

W1_b = [W1 b1];

%first layer
z1 = W1_b * X1_b';
a1 = sigmoid(z1);

if (p1~=0)
%     disp(['in p == ' num2str(p)]);
    mask1 = rand(size(a1));
    mask1 = (mask1 < p1) / (1-p1);
else
     mask1 = ones(size(a1));
end
  
d1 = a1 .* mask1;
d1_b = [d1;ones([1 N])];
W2_b = [W2 b2];
z2 = W2_b * d1_b;
a2 = sigmoid(z2);
   
if (p2 ~= 0)
    mask2 = rand(size(a2));
    mask2 = (mask2 < p2) / (1-p2);
else
    mask2 = ones(size(a2));
end

d2 = a2 .* mask2;
d2_b = [d2;ones([1 N])];
W3_b = [W3 b3];
z3 = W3_b * d2_b;


%%check for numerical stability
mat_max = repmat(max(z3),[output_dimension 1]);
z3 = z3 - mat_max;


end