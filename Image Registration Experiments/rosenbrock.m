function [f, df, d2f] = rosenbrock(x)
%--------------------------------------------------------------------------
% This file is part of my master's thesis entitled
%           'Low rank- and sparsity-based image registration'
% For the whole project see
%           https://github.com/roland1993/MA
% If you have questions contact me at
%           roland.haase [at] student.uni-luebeck [dot] de
% Source code is provided under the
%           MIT Open Source License
%--------------------------------------------------------------------------
% IN:
%   x   ~ 2 x 1         evaluation point
% OUT:
%   f   ~ 1 x 1         function value f(x)
%   df  ~ 2 x 1         gradient of f at x
%   d2f ~ 2 x 2         Hessian of f at x
%--------------------------------------------------------------------------

f = (1 - x(1)) ^ 2 + 100 * (x(2) - x(1) ^ 2) ^ 2;
df = [(-2) * (1 - x(1)) - 400 * x(1) * (x(2) - x(1) ^ 2); ...
    200 * (x(2) - x(1) ^ 2)];
d2f = [2 + 800 * x(1) ^ 2 - 400 * (x(2) - x(1) ^ 2), (-400) * x(1); ...
    (-400) * x(1), 200];

end