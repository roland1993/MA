function [x_star, y_star] = ...
    chambolle_pock(F, G, K, x0, y0, theta, tau, sigma, maxIter)
% Solve the primal minimization problem
%       min_x   F(Kx) + G(x) 
% by primal-dual reformulation into a saddle-point problem
%       min_x max_y <Kx,y> + G(x) - F*(y),
% where F, G are convex, proper and lsc. Note that both problems are 
% equivalent to the dual problem
%       max_y -G*(-K'y) - F*(y).
%
% IN:
%   F       ~ function handle       to return (based on switch flag)
%                                       - function value F(u)
%                                       - prox-operator of F
%                                           [(id + sigma dF)^(-1)](u)
%                                               OR
%                                       - convex conjugate value F*(u)
%                                       - prox-operator of F*
%                                           [(id + sigma dF*)^(-1)](u)
%   G       ~ function handle       (see F, where tau <-> sigma)
%   K       ~ m x n                 linear operator x-domain -> y-domain
%   x0      ~ n x 1                 starting point for primal variables
%   y0      ~ m x 1                 starting point for dual variables
%   theta   ~ 1 x 1                 algorithm parameter from [0, 1]               
%   tau     ~ 1 x 1                 primal step width
%   sigma   ~ 1 x 1                 dual step width
%       NOTE: tau * sigma * ||K||^2 < 1 to ensure convergence!
%   maxIter ~ 1 x 1                 max #iterations as stopping criterion
% OUT:
%   x_star  ~ n x 1                 minimizer of primal problem
%   y_star  ~ m x 1                 maximizer of dual problem

% set standard parameters
if nargin < 9, maxIter = 100; end
if nargin < 8
    
    % estimate squared spectral norm of K to determine sigma from tau
    K_abs = abs(K);
    L_squared_estimate = max(sum(K_abs, 1)) * max(sum(K_abs, 2));
    
    sigma = (1 - 1e-3) / (L_squared_estimate * tau);
    
end
if nargin < 7
    
    % estimate spectral norm of K to determine sigma from tau
    K_abs = abs(K);
    L_estimate = sqrt(max(sum(K_abs, 1)) * max(sum(K_abs, 2)));
    
    % if neither tau nor sigma were provided -> make them equal
    tau = (1 - 1e-3) / L_estimate;
    sigma = tau;
    
end
if nargin < 6, theta = 1; end

% initialize iteration counter
i = 0;

% initialize iteration variables
y_current = y0;
x_current = x0;
x_bar_current = x0;

% perform iteration
while (i < maxIter)
    
    % increase iteration counter
    i = i + 1
    
    % save old variables
    x_bar_old = x_bar_current;
    x_old = x_current;
    
    % get y_{n+1} = [(id + sigma dF*)^(-1)](y_n + sigma * K * x_bar_n)
    [~, y_current] = F(y_current + sigma * (K * x_bar_old));
    
    % get x_{n+1} = [(id + tau dG)^(-1)](x_n - tau * K' * y_{n+1})
    [~, x_current] = G(x_current - tau * (K' * y_current));
    
    % x_bar_{n+1} = x_{n+1} + theta * (x_{n+1} - x_n)
    x_bar_current = x_current + theta * (x_current - x_old);
    
end

% output last iterates
x_star = x_current;
y_star = y_current;

end