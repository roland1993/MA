function [x_star, x_history] = ...
    gradient_descent(f, x0, tol1, maxIter, tol2)
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
%   f           ~ function handle       target function, returns
%                                           - f(x)  ~ 1 x 1
%                                           - df/dx ~ k x 1
%   x0          ~ k x 1                 starting point
%   tol1        ~ 1 x 1                 tolerance for necessary condition
%   maxIter     ~ 1 x 1                 maximum number of iterations
%   tol2        ~ 1 x 1                 tolerance for target fctn. decrease
% OUT:
%   x_star      ~ k x 1                 minimizer of f / last iterate
%   x_history   ~ k x #iter             recording of all iterates of x
%--------------------------------------------------------------------------

% set standard parameters if not provided
if nargin < 5, tol2 = 1e-2; end
if nargin < 4, maxIter = 500; end
if nargin < 3, tol1 = 1e-3; end

% iteration counter
i = 0;

% tracking of target fctn. values
f_history = zeros(maxIter + 1, 1);

% evaluate starting point
x_current = x0;
[f_cur, df] = f(x_current);
f_history(1) = f_cur;

% return all iterates of x (if requested)
if nargout == 2
    x_history = zeros(length(x0), maxIter + 1);
    x_history(:, 1) = x0;
end

% output some info
fprintf('\nGRADIENT DESCENT ON \n\n\t%s\n\nSTOPPING CRITERIA\n\n', ...
    func2str(f));
fprintf('\tTOLERANCE ||grad(f)(x)|| <= %.1e\n', tol1);
fprintf('\tMAXITER = %d\n', maxIter);
fprintf('\tDECREASE OVER 5 ITERATES <= %.1e\n\n', tol2);
fprintf('i \t ||grad(f)(x_i)|| \t f(x_i)/f(x_i-1)\n');
fprintf('------------------------------------------------\n');

% gradient descent iteration
while (norm(df) > tol1) && ...
        (i < maxIter) && ...
        ((i < 5) || ((f_history(i + 1) / f_history(i - 4)) < (1 - tol2)))
    % 3rd stopping criterion: if the last 5 iterations did not manage to
    %   decrease f by at least (100 * tol2) percent -> stop iterating
    
    i = i + 1;
    dir = -df;
    
    % line search for step size alpha
    alpha = armijo(f, x_current, dir);
    
    % update current point
    x_current = x_current + alpha * dir;
    [f_cur, df] = f(x_current);
    f_history(i + 1) = f_cur;
    if nargout == 2
        x_history(:, i + 1) = x_current;
    end
    
    % output progress
    fprintf('%d \t %.2e \t\t %.4f\n', i, norm(df), f_cur / f_history(i));
end

x_star = x_current;
if nargout == 2
    x_history(:, (i + 2) : end) = [];
end

% final output
fprintf('\nSTOPPING AT CRITERION\n\n');
if (i == maxIter)
    fprintf('\t#iter = maxIter = %d\n', maxIter);
elseif (norm(df) <= tol1)
    fprintf('\t||grad(f)(x_i)|| = %.2e <= %.1e\n', norm(df), tol1);
else
    fprintf('\tDECREASE OVER LAST 5 ITERATES = %.1e <= %.1e\n', ...
        (1 - (f_history(i + 1) / f_history(i - 4))), tol2);
end
fprintf('\nREMAINDER OF INITIAL TARGET\n\n');
fprintf('\tf(x_star)/f(x_0) = %.4e\n\n', ...
    (f_history(i + 1) / f_history(1)));

end