%   min_{u,L} delta_{|| . || <= nu}(B * L)
%               + sum_i || l_i - I_i(u_i) ||_1
%               + mu * sum_i TV(u_i)
%
%   MEAN-FREE & NO REFERENCE

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

%% INITIALIZATION

% clean-up
clear all, clc, close all;

% make sure that interpolation routines are on search path
if ~exist('evaluate_displacement.m', 'file')
    addpath(genpath('..'));
end

% some local function definitions
vec = @(x) x(:);
normalize = @(x) (x - min(x(:))) / (max(x(:)) - min(x(:)));

% % ~~~~~~~ SYNTHETIC IMAGE DATA ~~~~~~~
% m = 32;
% n = 32;
% numFrames = 9;
% ex = 3;
% A = createTestImage(m, n, numFrames, ex);
% 
% k = 4;
% idx = ceil(numFrames / 2) - floor(k / 2) + (1 : k);
% img = cell(k, 1);
% for i = 1 : k
%     img{i} = normalize(A(:, :, idx(i)));
% end
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-2;
% outerIter = 15;
% mu = 1e0;
% nu_factor = 0.85;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% % ~~~~~~~ ROTATING STAR DATA ~~~~~~~
% k = 2;
% img{1} = normalize(double(imread('rotation_star1.png')));
% img{2} = normalize(double(imread('rotation_star2.png')));
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-2;
% outerIter = 10;
% mu = 1e-1;
% nu_factor = 0.8;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% % ~~~~~~~ SLIDING RECT DATA ~~~~~~~
% k = 2;
% img{1} = double(rgb2gray(imread('sr1.png')));
% img{1} = normalize(img{1});
% img{2} = double(rgb2gray(imread('sr2.png')));
% img{2} = normalize(img{2});
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-2;
% outerIter = 5;
% mu = 1e-1;
% nu_factor = 0.65;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% % ~~~~~~~~~ HEART MRI CINE ~~~~~~~~~
% k = 3;
% load('heart_mri.mat');
% IDX = floor(linspace(beats(1, 1), beats(1, 2), k + 1));
% factor = 4;
% for i = 1 : (k + 1)
%     % downsampling
%     tmp = conv2(data(:, :, IDX(i)), ones(factor) / factor ^ 2, 'same');
%     img{i} = tmp(1 : factor : end, 1 : factor : end);
% end
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-2;
% outerIter = 15;
% mu = 2e-2;
% nu_factor = 0.95;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% ~~~~~~~~~ DYNAMIC TEST IMAGES ~~~~~~~~~
m = 100;
n = 100;
k = 8;
data = dynamicTestImage(m, n, k);
img = cell(1, k);
for i = 1 : k, img{i} = data(:, :, i);  end

% optimization parameters - OPTIMIZED!
theta = 1;
maxIter = 2000;
tol = 1e-3;
outerIter = 20;
mu = 1e-1;
nu_factor = 0.9;
bc = 'linear';
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% get image resolution etc.
[m, n] = size(img{1});
h_img = [1, 1];
omega = [0, m, 0, n];
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% choose reference and template images
T = zeros(m, n, k);
for i = 1 : k
    T(:, :, i) = ...
        evaluate_displacement(img{i}, h_img, zeros(m * n, 2), omega);
end

%% OPTIMIZATION SCHEME

% initialize primary and dual variables
x = zeros(3 * k * m * n, 1);
p = zeros(6 * k * m * n, 1);

% gradient operator on displacements u
A2 = finite_difference_operator(m, n, h_grid, k, bc);

% all zeros
A3 = sparse(k * m * n, 2 * k * m * n);

% identity matrix
A4 = speye(k * m * n);

% all zeros
A5 = sparse(4 * k * m * n, k * m * n);

% mean free operator
A6 = mean_free_operator(m, n, k);

% prepare figures for output display
fh1 = figure(1);
set(fh1, 'units', 'normalized', 'outerposition', [0 0 1 1]);
fh2 = figure(2);
set(fh2, 'units', 'normalized', 'outerposition', [0 0 1 1]);
green = cat(3, zeros(m, n), ones(m, n), zeros(m, n));

% track change in singluar values
SV_history = zeros(k, outerIter);

for o = 1 : outerIter
%-------------------------------------------------------------------------%
% BEGIN OUTER ITERATION
    
    % fetch u-part from primary variables
    u0 = x(1 : 2 * k * m * n);
    u0 = reshape(u0, m * n, 2, k);
    
    % reference vector for computing SAD from L
    b = zeros(k * m * n, 1);
    
    % templates evaluated using current u
    T_current = zeros(m, n, k);
    
    % store image gradients of template images, complete vector b
    dT = cell(k, 1);
    for i = 1 : k
        [T_current(:, :, i), dT{i}] = ...
            evaluate_displacement(img{i}, h_img, u0(:, :, i), omega);
        b((i - 1) * m * n + 1 : i * m * n) = ...
            vec(T_current(:, :, i)) - dT{i} * vec(u0(:, :, i));
    end
    
    % estimate threshold nu from nuclear norm of mean free images
    if o == 1
        D = reshape(A6 * vec(T_current), m * n, k);
    else
        D = reshape(A6 * vec(L_star), m * n, k);
    end
    [~, S, ~] = svd(D, 'econ');
    nu = nu_factor * sum(diag(S));
    
    % upper left block of A ~> template image gradients
    A1 = -blkdiag(dT{:});
    
    % build up A from the computed blocks
    A = [       A1,     A4
                A2,     A5          
                A3,     A6          ];
    
    % estimate spectral norm of A
    norm_A_est = normest(A);
    
    % use estimated norm to get primal and dual step sizes
    tau = sqrt(0.95 / norm_A_est ^ 2);
    sigma = sqrt(0.95 / norm_A_est ^ 2);
    
    % get function handle to G-part of target function
    G_handle = @(x, c_flag) zero_function(x, c_flag);
    
    % get function handle to F-part of target function
    F_handle = @(y, c_flag) F(y, b, k, mu, nu, sigma, c_flag);
    
    % perform optimization
    [x, p, primal_history, dual_history] = chambolle_pock(...
        F_handle, G_handle, A, x, p, theta, tau, sigma, maxIter, tol);
    
    % ITERATIVE OUTPUT
    
    % plot primal & dual energies
    figure(fh1);
    clf;
    set(fh1, 'Name', sprintf('ITERATE %d OUT OF %d', o, outerIter));
    
    subplot(2, 2, 1);
    hold on;
    plot(primal_history(:, 1), 'LineWidth', 1.5);
    plot(dual_history(:, 1), 'LineWidth', 1.5);
    hold off;
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'primal energy', 'dual energy'}, ...
        'FontSize', 12, 'Location', 'SouthOutside', ...
        'Orientation', 'Horizontal');
    title('primal vs. dual')
    
    GAP = abs((primal_history(:, 1) - dual_history(:, 1)) ./ ...
        dual_history(:, 1));
    subplot(2, 2, 2);
    semilogy(GAP, 'LineWidth', 1.5);
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'Absolute normalized gap'}, 'FontSize', 12, ...
        'Location', 'SouthOutside', 'Orientation', 'Horizontal');
    title('primal-dual gap');
    
    subplot(2, 2, 3);
    semilogy(primal_history(:, 6));
    hold on;
    semilogy(primal_history(:, 7));
    semilogy(dual_history(:, 6));
    semilogy(dual_history(:, 7));
    hold off;
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'F', 'G', 'F*', 'G*'}, 'FontSize', 12, ...
        'Location', 'SouthOutside', 'Orientation', 'Horizontal');
    title('constraints');
    
    subplot(2, 2, 4);
    plot(primal_history(:, 1), '--' ,'LineWidth', 1.5);
    hold on;
    plot(primal_history(:, 2) ,'LineWidth', 1.5);
    plot(primal_history(:, 3) ,'LineWidth', 1.5);
    plot(primal_history(:, 4) ,'LineWidth', 1.5);
    hold off;
    axis tight;
    grid on;
    xlabel('#iter');
    legend(...
        {'F', '\Sigma_i || T_i(u_i) - l_i ||_1', '\Sigma_i TV (u_i)', ...
        '\delta_{|| . || <= \nu} (L - l_{mean})'}, ...
        'FontSize', 12, 'Location', 'SouthOutside', ...
        'Orientation', 'Horizontal');
    
    % evaluate minimizer x_star = [u_star; L_star]
    x_star = x;
    p_star = p;
    u_star = x_star(1 : 2 * k * m * n);
    u_star = reshape(u_star, m * n, 2, k);
    T_u = zeros(m, n, k);
    for i = 1 : k
        T_u(:, :, i) = ...
            evaluate_displacement(img{i}, h_img, u_star(:, :, i), omega);
    end
    
    L_star = x_star(2 * k * m * n + 1 : end);
    L_star = reshape(L_star, m, n, k);
    
    meanL = sum(L_star, 3) / k;
    [~, SV, ~] = svd(reshape(L_star - meanL, [], k), 'econ');
    SV_history(:, o) = vec(diag(SV));
    
    % get cell-centered grid over omega for plotting purposes
    [cc_x, cc_y] = cell_centered_grid(omega, [m, n]);
    cc_grid = [cc_x(:), cc_y(:)];
    
    % display resulting displacements & low rank components
    figure(fh2);
    clf;
    colormap gray(256);
    set(fh2, 'Name', sprintf('ITERATE %d OUT OF %d', o, outerIter));
    
    for i = 1 : k
        subplot(3, k, i);
        set(gca, 'YDir', 'reverse');
        imshow(T(:, :, i), [0, 1], 'InitialMagnification', 'fit');
        axis image;
        title(sprintf('template T_%d', i));
        hold on;
        quiver(cc_grid(:, 2), cc_grid(:, 1), ...
            u_star(:, 2, i), u_star(:, 1, i), 0, 'r');
        hold off;
    end
    
    for i = 1 : k
        subplot(3, k, k + i);
        set(gca, 'YDir', 'reverse');
        imshow(T_u(:, :, i), [0, 1], 'InitialMagnification', 'fit');
        axis image;
        title(sprintf('T_%d(u_%d)', i, i));
    end
    
    for i = 1 : k
        subplot(3, k, 2 * k + i);
        set(gca, 'YDir', 'reverse');
        imshow(L_star(:, :, i) - meanL, [-1, 1], ...
            'InitialMagnification', 'fit')
        axis image;
        title(sprintf('low rank component l_%d - l_{mean}', i));
    end
    
    drawnow;
    
%     % pause until button press
%     if o < outerIter
%         str = [sprintf('%s ITERATION #%d FINISHED %s\n\n', ...
%             repmat('~', [1, 10]), o, repmat('~', [1, 10])), ...
%             'PRESS ENTER TO CONTINUE: '];
%         input(str);
%     end
    
% END OUTER ITERATION
%-------------------------------------------------------------------------%
end

%% FINAL OUTPUT
% 
% figure;
% colormap gray(256);
% while true
%     for i = 1 : size(T, 3)
%         subplot(1, 2, 1);
%         imshow(T(:, :, i), [0, 1], 'InitialMagnification', 'fit');
%         title(sprintf('T_%d', i));
%         subplot(1, 2, 2);
%         imshow(T_u(:, :, i), [0, 1], 'InitialMagnification', 'fit');
%         title(sprintf('T_%d(u_%d)', i, i));
%         drawnow;
%         waitforbuttonpress;
%     end
% end

figure;
names = cell(k + 1, 1);
hold on;
for i = 1 : k
    plot(SV_history(i, :), '-x');
    names{i} = ['\sigma_', num2str(i)];
end
plot(sum(SV_history, 1), '--x');
names{k + 1} = '\Sigma_i \sigma_i';
hold off;
xlim([0.5, outerIter + 0.5]);
xlabel('#outer iter');
title('singular values of mean-free low-rank components');
grid on;
legend(names);

%% LOCAL FUNCTION DEFINITIONS

function [res1, res2, res3] = F(y, b, k, mu, nu, sigma, conjugate_flag)
% splits input y = [y1; y2; y3] and computes
%   F_1(y1) = || y1 - b ||_1
%   F_2(y2) = sum_i mu * || y2_i ||_{2,1}
%   F_3(y3) = delta_{|| . ||_* <= nu}(y3)

% get number of template images and number of pixels per image
mn = numel(y) / (6 * k);

% split input y into r- and v-part
y1 = y(1 : k * mn);
y2 = y(k * mn + 1 : 5 * k * mn);
y3 = y(5 * k * mn + 1 : end);

if nargout == 3
    
    % apply F1 = SAD to y1-part
    [~, ~, res3_F1] = SAD(y1, b, sigma, conjugate_flag);
    
    % initialize outputs with outputs from F1
    res3 = zeros(6 * k * mn, 1);
    res3(1 : k * mn) = res3_F1;
    
    % apply mu * ||.||_{2,1} to each of the k components y2_i of y2
    y2 = reshape(y2, 4 * mn, k);
    for i = 1 : k
        
        [~, ~, res3_F2] = norm21(y2(:, i), mu, sigma, conjugate_flag);
        % update outputs
        res3(k * mn + (i - 1) * 4 * mn + 1 : k * mn + i * 4 * mn) = ...
            res3_F2;
        
    end
    
    % apply delta_{|| . ||_* <= nu} to y3
    [~, ~, res3_F3] = ...
        nuclear_norm_constraint(y3, k, sigma, nu, conjugate_flag);
    
    % update outputs
    res3(5 * k * mn + 1 : end) = res3_F3;
    
    % dummy outputs
    res1 = [];
    res2 = [];
    
else
    
    % apply F1 = SAD to y1-part
    [res1_F1, res2_F1] = SAD(y1, b, sigma, conjugate_flag);
    
    % apply mu * ||.||_{2,1} to each of the k components y2_i of y2
    y2 = reshape(y2, 4 * mn, k);
    res1_F2 = 0;
    res2_F2 = 0;
    for i = 1 : k
        
        [res1_F2_i, res2_F2_i] = ...
            norm21(y2(:, i), mu, sigma, conjugate_flag);
        
        % update outputs
        res1_F2 = res1_F2 + res1_F2_i;
        res2_F2 = max(res2_F2, res2_F2_i);
        
    end
    
    % apply delta_{|| . ||_* <= nu} to y3
    [res1_F3, res2_F3] = ...
        nuclear_norm_constraint(y3, k, sigma, nu, conjugate_flag);
    
    % update outputs
    res1 = [res1_F1, res1_F2, res1_F3];
    res2 = max([res2_F1, res2_F2, res2_F3]);
    
end

end