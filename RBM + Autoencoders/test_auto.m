function []=test_auto(W,b,c)
  W2 = W';W1=W;
  W_c = [W2 c];
  W_b = [W1 b];
  
  x = randn([784,100]);
% x = val.images(:,100:200);

  x_1 = [x; ones([1 size(x,2)])];
  
[x_hat,~]=FP_auto(W_b,W_c,x_1);
im_weights = visualize(x_hat');
figure;
imshow(im_weights')
end