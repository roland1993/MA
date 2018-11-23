function [x_star, y_star] = ...
    chambolle_pock(F, G, K, x0, y0, theta, tau, sigma, ...
    maxIter, tol1, tol2, tol3)
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
%   G       ~ function handle       (see F, with tau <-> sigma)
%   K       ~ m x n                 linear operator x-domain -> y-domain
%   x0      ~ n x 1                 starting point for primal variables
%   y0      ~ m x 1                 starting point for dual variables
%   theta   ~ 1 x 1                 extragradient step parameter, in [0,1]
%   tau     ~ 1 x 1                 primal step width
%   sigma   ~ 1 x 1                 dual step width
%       NOTE: tau * sigma * ||K||^2 < 1 to ensure convergence!
%   maxIter ~ 1 x 1                 max #iterations as stopping criterion
%   tol1    ~ 1 x 1                 tolerance for duality gap change
%   tol2    ~ 1 x 1                 tolerance for primal objective change
%   tol3    ~ 1 x 1                 tolerance for dual objective change
% OUT:
%   x_star  ~ n x 1                 minimizer of primal problem
%   y_star  ~ m x 1                 maximizer of dual problem

% set standard parameters
if nargin < 12, tol3 = 1e-4; end
if nargin < 11, tol2 = 1e-4; end
if nargin < 10, tol1 = 1e-4; end
if nargin < 9, maxIter = 100; end
if nargin < 8
    % estimate squared spectral norm of K to determine sigma from tau
    K_abs = abs(K);
    L_squared_estimate = max(sum(K_abs, 1)) * max(sum(K_abs, 2));
    sigma = 1 / (L_squared_estimate * tau);
end
if nargin < 7
    % estimate spectral norm of K to determine sigma and tau
    K_abs = abs(K);
    L_estimate = sqrt(max(sum(K_abs, 1)) * max(sum(K_abs, 2)));
    % if neither tau nor sigma were provided -> make them equal
    tau = 1 / L_estimate;
    sigma = tau;
end
if nargin < 6, theta = 1; end

% initialize iteration counter
i = 0;

% initialize iteration variables
y_current = y0;
x_current = x0;
x_bar = x0;

% function handles for primal and dual objective
primal_objective = @(x) F(K*x, false) + G(x, false);
dual_objective = @(y) -G(-K'*y, true) - F(y, true);

% record progress in primal and dual objective
primal_history = zeros(maxIter + 1, 1);
dual_history = zeros(maxIter + 1, 1);
primal_history(1) = primal_objective(x0);
dual_history(1) = dual_objective(y0);

% output some info
fprintf('\nCHAMBOLLE POCK PRIMAL DUAL OPTIMIZATION SCHEME\n');
fprintf('\n\tPRIMAL PROBLEM:\t\tp(x) = F(Kx) + G(x)\t\t-> min!\n');
fprintf('\tDUAL PROBLEM:\t\tq(y) = -G*(-K*y) - F*(y)\t-> max!\n');
fprintf('\nFOR\n\n\tF = %s\n\tG = %s\n', func2str(F), func2str(G));
fprintf('\nWITH PARAMETERS\n');
fprintf('\n\tEXTRAGRADIENT STEP SIZE\t\tTHETA\t= %.3f', theta);
fprintf('\n\tPRIMAL STEP SIZE\t\tTAU\t= %.3f', tau);
fprintf('\n\tDUAL STEP SIZE\t\t\tSIGMA\t= %.3f\n', sigma);
fprintf('\n\tNUMBER OF PRIMAL VARIABLES\t %d', numel(x0));
fprintf('\n\tNUMBER OF DUAL VARIABLES\t %d\n', numel(y0));
fprintf('\n\tMAX NUMBER OF ITERATIONS\t %d\n', maxIter);
fprintf(['\ni\tp(x_i)\t\tp(x_i)/p(x_i-1)\t q(y_i)', ...
    '\t\tq(y_i)/q(y_i-1)\t p(x_i)-q(y_i)\n']);
fprintf([repmat('-', [1, 86]), '\n']);
fprintf('%d\t%.2e\t-\t\t %.2e\t-\t\t %.2e\n', ...
    0, primal_history(1), dual_history(1), ...
    primal_history(1) - dual_history(1));

% perform iteration
while (i < maxIter) && ...
        ((i < 4) || ...
        abs(1 - (primal_history(i) / primal_history(i - 3))) > tol2) && ...
        ((i < 4) || ...
        abs(1 - (primal_history(i) - dual_history(i)) / ...
        (primal_history(i - 3) - dual_history(i - 3))) > tol1) && ...
        ((i < 4) || ...
        abs(1 - (dual_history(i) / dual_history(i - 3))) > tol3)
    
    % increase iteration counter
    i = i + 1;
    
    % save old x iterate for later use
    x_old = x_current;
    
    % get y_{n+1} = [(id + sigma dF*)^(-1)](y_n + sigma * K * x_bar_n)
    [~, y_current] = F(y_current + sigma * (K * x_bar), true);
    
    % get x_{n+1} = [(id + tau dG)^(-1)](x_n - tau * K' * y_{n+1})
    [~, x_current] = G(x_current - tau * (K' * y_current), false);
    
    % record primal and dual objective value for current iterates
    primal_history(i + 1) = primal_objective(x_current);
    dual_history(i + 1) = dual_objective(y_current);
    
    % x_bar_{n+1} = x_{n+1} + theta * (x_{n+1} - x_n)
    x_bar = x_current + theta * (x_current - x_old);
    
    % iterative output
    
    fprintf('%d\t%.2e\t%.4f\t\t %.2e\t%.4f\t\t %.2e\n', i, ...
        primal_history(i + 1), ...
        primal_history(i + 1) / primal_history(i), ...
        dual_history(i + 1), ...
        dual_history(i + 1) / dual_history(i), ...
        primal_history(i + 1) - dual_history(i + 1));
    
end

% more output
fprintf('\nSTOPPING AT CRITERION:\n\n\t');
if i == maxIter
    fprintf(...
        'MAX NUMBER OF ITERATIONS REACHED\ti = %d', ...
        maxIter);
    
elseif abs(1 - (primal_history(i) - dual_history(i)) / ...
        (primal_history(i - 3) - dual_history(i - 3))) <= tol1
    fprintf(...
        'RELATIVE GAP CHANGE OVER LAST 3 ITERATES %.2e <= %.2e', ...
        abs(1 - (primal_history(i) - dual_history(i)) / ...
        (primal_history(i - 3) - dual_history(i - 3))), tol1);
    
elseif abs(1 - (primal_history(i) / primal_history(i - 3))) <= tol2
    fprintf(...
        'RELATIVE PRIMAL CHANGE OVER LAST 3 ITERATES %.2e <= %.2e', ...
        abs(1 - (primal_history(i) / primal_history(i - 3))), tol2);
    
elseif abs(1 - (dual_history(i) / dual_history(i - 3))) <= tol3
    fprintf(...
        'RELATIVE DUAL CHANGE OVER LAST 3 ITERATES %.2e <= %.2e', ...
        abs(1 - (dual_history(i) / dual_history(i - 3))), tol3);
    
end
fprintf('\n\n');

% return last iterates
x_star = x_current;
y_star = y_current;

end