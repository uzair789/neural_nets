function [x_hat,h]=FP_auto(W_b,W_c,x_1)
     a1 = W_b*x_1;
     h = sigmoid(a1) ;
     h_1 = [h; ones([1 size(h,2)])];
     a2 = W_c * h_1;
     x_hat = sigmoid(a2);
end