function [x_star, y_star, primal_history, dual_history] = ...
    chambolle_pock(F, G, K, x0, y0, theta, tau, sigma, maxIter, tol)
%
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
%   tol     ~ 1 x 1                 tolerance for normalized duality gap
% OUT:
%   x_star          ~ n x 1         minimizer of primal problem
%   y_star          ~ m x 1         maximizer of dual problem
%   primal_history  ~ #iter x 1     history of primal energy over iterates
%   dual_history    ~ #iter x 1     history of dual energy over iterates

% set standard parameters
if nargin < 10, tol = 1e-3; end
if nargin < 9, maxIter = 300; end
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

% record progress in primal and dual objective
primal_history = zeros(maxIter + 1, 1);
dual_history = zeros(maxIter + 1, 1);
[primal_history(1), F_con, G_con] = primal_objective(x0);
[dual_history(1), FS_con, GS_con] = dual_objective(y0);

% output some info
fprintf('\nCHAMBOLLE POCK PRIMAL DUAL OPTIMIZATION SCHEME\n');
fprintf('\n\tPRIMAL PROBLEM\tp(x) = F(Kx) + G(x)\t\t-> min!\n');
fprintf('\tDUAL PROBLEM\tq(y) = -G*(-K*y) - F*(y)\t-> max!\n');
fprintf('\nFOR\n\n\tF = %s\n\tG = %s\n', func2str(F), func2str(G));
fprintf('\tGAP(x,y) = |p(x)-q(y)| / |q(y)|\n');
fprintf('\nWITH PARAMETERS\n');
fprintf('\n\tEXTRAGRADIENT STEP SIZE\t\tTHETA\t= %.3f', theta);
fprintf('\n\tPRIMAL STEP SIZE\t\tTAU\t= %.3f', tau);
fprintf('\n\tDUAL STEP SIZE\t\t\tSIGMA\t= %.3f\n', sigma);
fprintf('\n\tNUMBER OF PRIMAL VARIABLES\t %d', numel(x0));
fprintf('\n\tNUMBER OF DUAL VARIABLES\t %d\n', numel(y0));
fprintf('\n\tMAX NUMBER OF ITERATIONS\t %d', maxIter);
fprintf('\n\tTOLERANCE FOR NORMALIZED GAP\t %.1e\n', tol);
fprintf('\ni\tp(x_i)\t\tq(y_i)\t\tGAP(x_i,y_i)\tCONSTRAINTS HURT\n');
fprintf([repmat('-', [1, 72]), '\n']);
fprintf('%d\t%+.2e\t%+.2e\t%.3e', ...
    0, primal_history(1), dual_history(1), ...
    abs((primal_history(1) - dual_history(1)) / dual_history(1)));
if F_con > 1e-15, fprintf('\tF: %.2e', F_con); end
if G_con > 1e-15, fprintf('\tG: %.2e', G_con); end
if FS_con > 1e-15, fprintf('\tF*: %.2e', FS_con); end
if GS_con > 1e-15, fprintf('\tG*: %.2e', GS_con); end
fprintf('\n');

% perform iteration
while true
    
    % first stopping criterion
    if i == maxIter
        break;
    end
    
    % second stopping criterion
    GAP = abs((primal_history(i + 1) - dual_history(i + 1)) / ...
        dual_history(i + 1));
    if ~isnan(GAP) && (GAP <= tol)
        break;
    end
    
    % increase iteration counter
    i = i + 1;
    
    % save old x iterate for later use
    x_old = x_current;
    
    % get y_{n+1} = [(id + sigma dF*)^(-1)](y_n + sigma * K * x_bar_n)
    [~, y_current] = F(y_current + sigma * (K * x_bar), true);
    
    % get x_{n+1} = [(id + tau dG)^(-1)](x_n - tau * K' * y_{n+1})
    [~, x_current] = G(x_current - tau * (K' * y_current), false);
    
    % record primal and dual objective value for current iterates
    [primal_history(i + 1), F_con, G_con] = primal_objective(x_current);
    [dual_history(i + 1), FS_con, GS_con] = dual_objective(y_current);
    
    % x_bar_{n+1} = x_{n+1} + theta * (x_{n+1} - x_n)
    x_bar = x_current + theta * (x_current - x_old);
    
    % iterative output
    fprintf('%d\t%+.2e\t%+.2e\t%.3e', i, primal_history(i + 1), ...
        dual_history(i + 1), abs((primal_history(i + 1) - ...
        dual_history(i + 1)) / dual_history(i + 1)));
    if F_con > 1e-15, fprintf('\tF: %.2e', F_con); end
    if G_con > 1e-15, fprintf('\tG: %.2e', G_con); end
    if FS_con > 1e-15, fprintf('\tF*: %.2e', FS_con); end
    if GS_con > 1e-15, fprintf('\tG*: %.2e', GS_con); end
    fprintf('\n');

end

% more output
fprintf('\nSTOPPING AT CRITERION:\n\n\t');
if i == maxIter
    fprintf('MAX NUMBER OF ITERATIONS REACHED\ti = %d\n\n', maxIter);
else
    fprintf(['TOLERANCE FOR NORMALIZED GAP REACHED\t', ...
        '%.2e <= %.2e\n\n'], ...
        abs((primal_history(i + 1) - dual_history(i + 1)) / ...
        dual_history(i + 1)), tol);
    primal_history(i + 2 : end) = [];
    dual_history(i + 2 : end) = [];
end

% return last iterates
x_star = x_current;
y_star = y_current;

% primal and dual objective handles
    function [primal, F_constraint, G_constraint] = primal_objective(x)
        [F_val, ~, F_constraint] = F(K*x, false);
        [G_val, ~, G_constraint] = G(x, false);
        primal = F_val + G_val;
    end

    function [dual, FStar_constraint, GStar_constraint] = dual_objective(y)
        [FStar_val, ~, FStar_constraint] = F(y, true);
        [GStar_val, ~, GStar_constraint] = G(-K'*y, true);
        dual = (-1) * (FStar_val + GStar_val);
    end

end