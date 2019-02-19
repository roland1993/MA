function [u, L, primal_history, dual_history] = ...
    mf_nn_registration(img, refIdx, optPara)

% make sure that interpolation routines are on search path
if ~exist('evaluate_displacement.m', 'file')
    addpath(genpath('..'));
end

% some local function handles
vec = @(x) x(:);
normalize = @(x) (x - min(x(:))) / (max(x(:)) - min(x(:)));

% get number of temples images and corresponding indices
k = length(img) - 1;
tempIdx = setdiff(1 : k + 1, refIdx);

% normalize images to range [0, 1]
for i = 1 : (k + 1), img{i} = normalize(img{i}); end

% get reference image, image resolution etc.
R = img{refIdx};
[m, n] = size(R);
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
L = cell(outerIter, 1);
primal_history = cell(outerIter, 1);
dual_history = cell(outerIter, 1);

% initialize primal and dual variables
x = zeros((3 * k + 1) * m * n, 1);
p = zeros((6 * k + 2) * m * n, 1);

% gradient operator for displacement fields
A2 = finite_difference_operator(m, n, h_grid, k, bc);

% all zeros
A3 = sparse((k + 1) * m * n, 2 * k * m * n);

% identity matrix
A4 = speye((k + 1) * m * n);

% all zeros
A5 = sparse(4 * k * m * n, (k + 1) * m * n);

% mean free operator
A6 = mean_free_operator(m, n, k + 1);

% set function handle to G-part of target function
G_handle = @(x, c_flag) zero_function(x, c_flag);

% if plotting was requested -> create figure (handle)
if doPlots, fh = figure; end

%--------------OUTER ITERATION--------------------------------------------%
for o = 1 : outerIter
    
    % use computed u as u0 for new iterate
    if o == 1
        u0 = zeros(m * n, 2, k);
    else
        u0 = u{o - 1};
    end
    
    % reference vector for computing SAD from L
    b = zeros((k + 1) * m * n, 1);
    
    % set b-entries corresponding to reference
    b((k * m * n + 1) : end) = vec(R);
    
    % templates evaluated using current u0
    T_u = zeros(m, n, k);
    dT = cell(k, 1);
    for i = 1 : k
        [T_u(:, :, i), dT{i}] = evaluate_displacement( ...
            img{tempIdx(i)}, h_img, u0(:, :, i));
        b((i - 1) * m * n + 1 : i * m * n) = ...
            vec(T_u(:, :, i)) - dT{i} * vec(u0(:, :, i));
    end
    
    % get low-rank image-matrix
    if o == 1
        L_star = [vec(T_u); vec(R)]; 
    else
        L_star = vec(L{o - 1});
    end
    
    % estimate new threshold nu from nn of low-rank mean-free image-matrix
    D = reshape(A6 * L_star, m * n, k + 1);
    [~, S, ~] = svd(D, 'econ');
    nu = nu_factor * sum(diag(S));
    
    % upper left block of A ~> template image gradients
    A1 = [          -blkdiag(dT{:})
                sparse(m * n, 2 * k * m * n)    ];
    
    % build up A from the computed blocks
    A = [       A1,     A4
                A2,     A5          
                A3,     A6          ];
    
    % estimate spectral norm of A
    norm_A_est = normest(A);
    
    % use estimated norm to get primal and dual step sizes
    tau = sqrt(0.975 / norm_A_est ^ 2);
    sigma = sqrt(0.975 / norm_A_est ^ 2);
    
    % get function handle to F-part of target function
    F_handle = @(y, c_flag) F(y, b, k, mu, nu, sigma, c_flag);
    
    % perform optimization
    [x, p, primal_history{o}, dual_history{o}] = chambolle_pock( ...
        F_handle, G_handle, A, x, p, theta, tau, sigma, maxIter, tol);
    
    % get displacements and low rank components from (primal) minimizer x
    u{o} = reshape(x(1 : 2 * k * m * n), m * n, 2, k);
    L{o} = reshape(x(2 * k * m * n + 1 : end), m, n, k + 1);
    
    % plot progress
    if doPlots, plot_progress; end

end
%-------------------------------------------------------------------------%

%-------LOCAL FUNCTION DEFINITIONS----------------------------------------%
    function [res1, res2, res3] = ...
            F(y, b, k, mu, nu, sigma, conjugate_flag)
        % splits input y = [y1; y2; y3] and computes
        %   F_1(y1) = || y1 - b ||_1
        %   F_2(y2) = sum_i mu * || y2_i ||_{2,1}
        %   F_3(y3) = delta_{|| . ||_* <= nu}(y3)
        
        % get number of template images and number of pixels per image
        mn = numel(y) / (6 * k + 2);
        
        % split input y into r- and v-part
        y1 = y(1 : (k + 1) * mn);
        y2 = y((k + 1) * mn + 1 : (5 * k + 1) * mn);
        y3 = y((5 * k + 1) * mn + 1 : end);
        
        if nargout == 3
            
            % initialize output
            res3 = zeros((6 * k + 2) * mn, 1);
            
            % apply SAD to y1-part
            [~, ~, res3_F1] = SAD(y1, b, sigma, conjugate_flag);
            res3(1 : (k + 1) * mn) = res3_F1;
            
            % apply mu * ||.||_{2,1} to each of the k components y2_i
            y2 = reshape(y2, 4 * mn, k);
            for j = 1 : k
                [~, ~, res3_F2] = ...
                    norm21(y2(:, j), mu, sigma, conjugate_flag);
                res3((k + 1) * mn + (j - 1) * 4 * mn + 1 : ...
                    (k + 1) * mn + j * 4 * mn) = res3_F2;
            end
            
            % apply delta_{|| . ||_* <= nu} to y3
            [~, ~, res3_F3] = nuclear_norm_constraint( ...
                y3, k + 1, sigma, nu, conjugate_flag);
            res3((5 * k + 1) * mn + 1 : end) = res3_F3;
            
            % dummy outputs
            res1 = [];
            res2 = [];
            
        else
            
            % apply F1 = SAD to y1-part
            [res1_F1, res2_F1] = SAD(y1, b, sigma, conjugate_flag);
            
            % apply mu * ||.||_{2,1} to each of the k components y2_i
            y2 = reshape(y2, 4 * mn, k);
            res1_F2 = 0;
            res2_F2 = 0;
            for j = 1 : k
                [res1_F2_i, res2_F2_i] = ...
                    norm21(y2(:, j), mu, sigma, conjugate_flag);
                res1_F2 = res1_F2 + res1_F2_i;
                res2_F2 = max(res2_F2, res2_F2_i);
            end
            
            % apply delta_{|| . ||_* <= nu} to y3
            [res1_F3, res2_F3] = nuclear_norm_constraint( ...
                y3, k + 1, sigma, nu, conjugate_flag);
            
            % compute outputs
            res1 = [res1_F1, res1_F2, res1_F3];
            res2 = max([res2_F1, res2_F2, res2_F3]);
            
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
        subplot(2, 2, 1);
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
        subplot(2, 2, 2);
        semilogy(GAP);
        axis tight;
        grid on;
        xlabel('#iter');
        legend({'absolute primal-dual gap'}, ...
            'Location', 'SouthOutside', ...
            'Orientation', 'Horizontal');
        title('primal-dual gap');
        
        % plot constraints
        subplot(2, 2, 3);
        semilogy(primal_history{o}(:, 6));
        hold on;
        semilogy(primal_history{o}(:, 7));
        semilogy(dual_history{o}(:, 6));
        semilogy(dual_history{o}(:, 7));
        hold off;
        axis tight;
        grid on;
        xlabel('#iter');
        legend({'F', 'G', 'F*', 'G*'}, ...
            'Location', 'SouthOutside', 'Orientation', 'Horizontal');
        title('constraints');
        
        % plot different components of F
        subplot(2, 2, 4);
        plot(primal_history{o}(:, 1));
        hold on;
        plot(primal_history{o}(:, 2));
        plot(primal_history{o}(:, 3));
        hold off;
        axis tight;
        ylim([0, max(primal_history{o}(:, 1))]);
        grid on;
        xlabel('#iter');
        legend({'F', '\Sigma_i || T_i(u_i) - l_i ||_1', ...
            '\Sigma_i TV(u_i)'}, 'Location', 'SouthOutside', ...
            'Orientation', 'Horizontal');
        title('decomposition of F');
        
        drawnow;
        
    end
%-------------------------------------------------------------------------%

end