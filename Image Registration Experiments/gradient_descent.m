function [x_star, success, x_history] = ...
    gradient_descent(f, x0, tol, maxIter)
% IN:
%   f       ~ function handle       target function to minimize
%                                   (returns f(x) ~ 1 x 1, df/dx ~ k x 1)
%   x0      ~ k x 1                 starting point
%   tol     ~ 1 x 1                 tolerance for necessary condition
%   maxIter ~ 1 x 1                 maximum number of iterations
% OUT:
%   x_star  ~ k x 1                 minimizer of f / last iterate
%   success ~ logical               was a minimizer found?

% iteration counter
i = 0;

% evaluate starting point
x_current = x0;
[f_cur, df] = f(x_current);

% return all iterates of x if requested
if nargout == 3
    x_history = zeros(length(x0), maxIter + 1);
    x_history(:, 1) = x0;
end

% output progress
fprintf('i \t ||grad(f)(x_i)|| \t f(x_i)/f(x_i-1)\n');
fprintf('------------------------------------------------\n');

% gradient descent iteration
while (norm(df) > tol) && (i < maxIter)
    
    f_old = f_cur;
    i = i + 1;
    dir = -df;
    
    % line search for step size alpha
    alpha = armijo(f, x_current, dir);
    
    % update current point
    x_current = x_current + alpha * dir;
    [f_cur, df] = f(x_current);
    if nargout == 3
        x_history(:, i + 1) = x_current;
    end
    
    % output progress
    fprintf('%d \t %.2e \t\t %.4f\n', i, norm(df), f_cur / f_old);
end

x_star = x_current;
success = (i <= maxIter);
if nargout == 3
    x_history(:, (i + 2) : end) = [];
end

end