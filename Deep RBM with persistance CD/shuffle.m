function [x_shufled] = shuffle(x)
[row,col] = size(x);
pos = randperm(col);
x_shufled = x(:,pos);
end