function alpha = armijo(f, x, dir, alpha0, beta, tau)
% IN:
%   f       ~ function handle       target function to minimize
%   x       ~ k x 1                 current point
%   dir     ~ k x 1                 descent direction at point x
%   alpha0  ~ 1 x 1                 maximum step size
%   beta    ~ 1 x 1                 relative expected reduction from (0, 1)
%   tau     ~ 1 x 1                 step size shrinkage parameter
% OUT:
%   alpha   ~ 1 x 1                 step size

% set parameters to standard values if not specified
if nargin < 6, tau = 0.5; end
if nargin < 5, beta = 0.01; end
if nargin < 4, alpha0 = 1; end

% evaluate f and grad_f at x
alpha = alpha0;
[f_x, grad_f] = f(x);

% shrink step size alpha until Armijo condition is fulfilled
while ~(f(x + alpha * dir) <= f_x + beta * alpha * grad_f' * dir)
    alpha = tau * alpha;
end

end