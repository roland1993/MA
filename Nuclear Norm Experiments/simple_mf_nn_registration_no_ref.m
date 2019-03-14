%   min_{u,L} delta_{|| . || <= nu}(B * [I_1(u_1), .., I_k(u_k), R])
%       + sum_i mu * TV(u_i) + delta_{mean(u_x) = 0, mean(u_y) = 0}(u)
%
%   MEAN-FREE & NO REFERENCE & UNIQUENESS-TERM

function [u, primal_history, dual_history] = ...
    simple_mf_nn_registration_no_ref(img, optPara)
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
%   img     ~ cell(k, 1)        array of images
%   refIdx  ~ 1 x 1             index of reference image inside of img
%   optPara ~ struct            optimization parameters with fields
%       .theta      ~ 1 x 1     over-relaxation parameter
%       .maxIter    ~ 1 x 1     maximum number of iterations
%       .tol        ~ 1 x 1     tolerance for p/d-gap + infeasibilities
%       .outerIter  ~ 1 x 1     number of outer iterations
%       .mu         ~ 1 x 1     weighting factor (see model above)
%       .nu_factor  ~ 1 x 1     reduction of nu per outer iterate
%       .bc         ~ string    boundary condition for grid discretization
%       .doPlots    ~ logical   do plots during optimization?
%
% OUT:                                      PER OUTER ITERATE:
%   u               ~ cell(outerIter, 1)        displacement fields
%   L               ~ cell(outerIter, 1)        low rank components
%   primal_history  ~ cell(outerIter, 1)        primal iteration history
%   dual_history    ~ cell(outerIter, 1)        dual iteration history
%--------------------------------------------------------------------------

% make sure that interpolation routines are on search path
if ~exist('evaluate_displacement.m', 'file')
    addpath(genpath('..'));
end

% some local function handles
vec = @(x) x(:);
normalize = @(x) (x - min(x(:))) / (max(x(:)) - min(x(:)));

% get number of temples images and corresponding indices
k = length(img);

% normalize images to range [0, 1]
for i = 1 : k, img{i} = normalize(img{i}); end

% get image resolution etc.
[m, n] = size(img{1});
h_img = [1, 1];
omega = [0, m, 0, n];
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% set optimization parameters
theta = optPara.theta;
maxIter = optPara.maxIter;
tol = optPara.tol;
outerIter = optPara.outerIter;
mu = optPara.mu;
nu_factor = optPara.nu_factor;
bc = optPara.bc;
doPlots = optPara.doPlots;

% initialize outputs
u = cell(outerIter, 1);
primal_history = cell(outerIter, 1);
dual_history = cell(outerIter, 1);

% initialize primal and dual variables
x = zeros(2 * k * m * n, 1);
p = zeros(5 * k * m * n, 1);

% get mean free operator
M = mean_free_operator(m, n, k);

% gradient operator for displacement fields
A2 = finite_difference_operator(m, n, h_grid, k, bc);

% set function handle to G-part of target function
G_handle = @(x, c_flag) mean_zero_indicator(x, [m, n, k], c_flag);

% if plotting was requested -> create figure (handle)
if doPlots, fh = figure; end

%--------------OUTER ITERATION--------------------------------------------%
for o = 1 : outerIter
    
    % use computed u as u0 for new iterate
    if o == 1
        u0 = reshape(x, m * n, 2, k);
    else
        u0 = u{o - 1};
    end
    
    % constant offset vector for nuclear norm
    b = zeros(k * m * n, 1);
    
    % templates evaluated using current u0
    T_u = zeros(m, n, k);
    dT = cell(k, 1);
    for i = 1 : k
        [T_u(:, :, i), dT{i}] = evaluate_displacement( ...
            img{i}, h_img, u0(:, :, i));
        b((i - 1) * m * n + 1 : i * m * n) = ...
            vec(T_u(:, :, i)) - dT{i} * vec(u0(:, :, i));
    end
    
    % upper block of A ~> template image gradients
    A1 = M * blkdiag(dT{:});
    
    % build up A from the computed blocks
    A = [   	A1
                A2      ];
    
    % estimate spectral norm of A
    e = matrix_norm(A);
    norm_A_est = e(end);
    
    % use estimated norm to get primal and dual step sizes
    tau = sqrt(0.99 / norm_A_est ^ 2);
    sigma = sqrt(0.99 / norm_A_est ^ 2);
    
    % estimate nu-parameter
    I = reshape(M * vec(T_u), m * n, k);
    [~, SV, ~] = svd(I, 'econ');
    nu = nu_factor * sum(diag(SV));
    
    % get function handle to F-part of target function
    b = M * (-b);
    F_handle = @(y, c_flag) F(y, b, k, nu, mu, sigma, c_flag);
    
    % perform optimization
    [x, p, primal_history{o}, dual_history{o}] = chambolle_pock( ...
        F_handle, G_handle, A, x, p, theta, tau, sigma, maxIter, tol);
    
    % get displacements u from (primal) minimizer x
    u{o} = reshape(x, m * n, 2, k);
    
    % plot progress
    if doPlots, plot_progress; end
    
end
%-------------------------------------------------------------------------%

%-------LOCAL FUNCTION DEFINITIONS----------------------------------------%
    function [res1, res2, res3] = ...
            F(y, d, k, nu, mu, sigma, conjugate_flag)
        % splits input y = [v; w; x] and computes
        %   F_1(v) = delta_{|| . ||_* <= nu}(v - d)
        %   F_2(w) = sum_i mu * || w_i ||_{2,1}
        
        % get number of pixels per image
        mn = numel(y) / (5 * k);
            
        % split input y into v- and w-part
        v = y(1 : k * mn);
        w = y(k * mn + 1 : 5 * k * mn);
        
        % for the sake of efficiency: compute prox only if requested!
        if nargout == 3
            
            % apply nuclear norm to v-part
            [~, ~, res3_F1] = nuclear_norm_constraint_mod( ...
                v, d, k, sigma, nu, conjugate_flag);
            
            % initialize outputs with values from F1
            res3 = zeros(5 * k * mn, 1);
            res3(1 : k * mn) = res3_F1;
            
            % apply mu * ||.||_{2,1} to each of the k components v_i of v
            w = reshape(w, 4 * mn, k);
            for j = 1 : k
                [~, ~, res3_F2] = ...
                    norm21(w(:, j), mu, sigma, conjugate_flag);
                % update outputs
                res3(k * mn + (j - 1) * 4 * mn + 1 : ...
                    k * mn + j * 4 * mn) = res3_F2;
            end
            
            % dummy outputs
            res1 = [];
            res2 = [];
            
        else
            
            % apply nuclear norm to v-part
            [res1_F1, res2_F1] = nuclear_norm_constraint_mod( ...
                v, d, k, sigma, nu, conjugate_flag);
            
            % initialize outputs with values from F1
            res1 = res1_F1;
            res2 = res2_F1;
            
            % apply mu * ||.||_{2,1} to each of the k components v_i of v
            w = reshape(w, 4 * mn, k);
            for j = 1 : k
                [res1_F2, res2_F2] = ...
                    norm21(w(:, j), mu, sigma, conjugate_flag);
                % update outputs
                res1 = res1 + res1_F2;
                res2 = max(res2, res2_F2);
            end
            
        end
        
    end
%-------------------------------------------------------------------------%
    function plot_progress
        % plots progress of one outer iterate
        
        % make figure active
        figure(fh);
        set(fh, 'NumberTitle', 'off', ...
            'Name', sprintf('ITERATE %d OUT OF %d', o, outerIter));
        
        % plot primal vs. dual energy
        subplot(1, 3, 1);
        plot(primal_history{o}(:, 1));
        hold on;
        plot(dual_history{o}(:, 1));
        hold off;
        axis tight;
        grid on;
        xlabel('#iter');
        legend({'primal energy', 'dual energy'}, ...
            'Location', 'SouthOutside', ...
            'Orientation', 'Horizontal');
        title('primal vs. dual')
        
        % plot numerical gap
        GAP = abs((primal_history{o}(:, 1) - dual_history{o}(:, 1)) ./ ...
            dual_history{o}(:, 1));
        subplot(1, 3, 2);
        semilogy(GAP);
        axis tight;
        grid on;
        xlabel('#iter');
        legend({'absolute primal-dual gap'}, ...
            'Location', 'SouthOutside', ...
            'Orientation', 'Horizontal');
        title('primal-dual gap');
        
        % plot constraints
        subplot(1, 3, 3);
        semilogy(primal_history{o}(:, 4));
        hold on;
        semilogy(primal_history{o}(:, 5));
        semilogy(dual_history{o}(:, 4));
        semilogy(dual_history{o}(:, 5));
        hold off;
        axis tight;
        grid on;
        xlabel('#iter');
        legend({'F', 'G', 'F*', 'G*'}, ...
            'Location', 'SouthOutside', 'Orientation', 'Horizontal');
        title('constraints');
        
        drawnow;
        
    end
%-------------------------------------------------------------------------%

end