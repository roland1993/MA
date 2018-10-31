function [f, df] = rosenbrock(x)
% IN:
%   x ~ 2 x 1           evaluation point
% OUT:
%   f  ~ 1 x 1          function value f(x)
%   df ~ 2 x 1          gradient of f at x

f = (1 - x(1)) ^ 2 + 100 * (x(2) - x(1) ^ 2) ^ 2;
df = [(-2) * (1 - x(1)) - 400 * x(1) * (x(2) - x(1) ^ 2); ...
    200 * (x(2) - x(1) ^ 2)];

end