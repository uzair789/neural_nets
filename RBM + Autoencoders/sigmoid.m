function [x] = sigmoid(t)

x = 1 ./ (1 + exp(-t));

end