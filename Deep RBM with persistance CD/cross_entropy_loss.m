function [l] = cross_entropy_loss(x,x_hat)
    ii1 = x .* log(x_hat) + (1 - x) .* log(1 - x_hat) ;
    l = -sum(sum(ii1));
end