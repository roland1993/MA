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
% nu_factor = 0.1;
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
% nu_factor = 0.1;
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
% nu_factor = 0.1;
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
% nu_factor = 0.4;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% ~~~~~~~~~ DYNAMIC TEST IMAGES ~~~~~~~~~
m = 100;
n = 100;
k = 6;
data = dynamicTestImage(m, n, k);
for i = 1 : k, img{i} = data(:, :, i);  end

% optimization parameters
theta = 1;
maxIter = 1000;
tol = 1e-2;
outerIter = 15;
mu = 1e-1;
nu_factor = 0.2;
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
    D = reshape(A6 * vec(T_current), m * n, k);
    [~, S, ~] = svd(D, 'econ');
    nu = (nu_factor ^ (1 / outerIter)) * sum(diag(S));
    
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
    
    %% ITERATIVE OUTPUT
    
    % plot primal & dual energies
    figure(fh1);
    clf;
    set(fh1, 'Name', sprintf('ITERATE %d OUT OF %d', o, outerIter));
    
    subplot(2, 3, 1);
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
    subplot(2, 3, 2);
    semilogy(GAP, 'LineWidth', 1.5);
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'Absolute normalized gap'}, 'FontSize', 12, ...
        'Location', 'SouthOutside', 'Orientation', 'Horizontal');
    title('primal-dual gap');
    
    subplot(2, 3, 3);
    semilogy(primal_history(:, 4));
    hold on;
    semilogy(primal_history(:, 5));
    semilogy(dual_history(:, 4));
    semilogy(dual_history(:, 5));
    hold off;
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'F', 'G', 'F*', 'G*'}, 'FontSize', 12, ...
        'Location', 'SouthOutside', 'Orientation', 'Horizontal');
    title('constraints');
    
    subplot(2, 3, 4);
    hold on;
    plot(primal_history(:, 1), 'LineWidth', 1.5);
    plot(primal_history(:, 2), '--', 'LineWidth', 1.5);
    plot(primal_history(:, 3), '--', 'LineWidth', 1.5);
    hold off;
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'F(Kx) + G(x)', 'F(Kx)', 'G(x)'}, ...
        'FontSize', 12, 'Location', 'SouthOutside', ...
        'Orientation', 'Horizontal');
    title('primal objective');
    
    subplot(2, 3, 5);
    hold on;
    plot(dual_history(:, 1), 'LineWidth', 1.5);
    plot(-dual_history(:, 2), '--', 'LineWidth', 1.5);
    plot(-dual_history(:, 3), '--', 'LineWidth', 1.5);
    hold off;
    axis tight;
    grid on;
    xlabel('#iter');
    legend({'-[F*(y) + G*(-K*y)]', '-F*(y)', '-G*(-K*y)'}, ...
        'FontSize', 12, 'Location', 'SouthOutside', ...
        'Orientation', 'Horizontal');
    title('dual objective');
    
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
    
    meanL = sum(L_star, 3) / k;
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

% apply F1 = SAD to y1-part
[res1_F1, res2_F1, res3_F1] = SAD(y1, b, sigma, conjugate_flag);

% initialize outputs with outputs from F1
res1 = res1_F1;
res2 = res2_F1;
res3 = zeros(6 * k * mn, 1);
res3(1 : k * mn) = res3_F1;

% apply mu * ||.||_{2,1} to each of the k components y2_i of y2
y2 = reshape(y2, 4 * mn, k);
for i = 1 : k
    
    [res1_F2, res2_F2, res3_F2] = ...
        norm21(y2(:, i), mu, sigma, conjugate_flag);
    
    % update outputs
    res1 = res1 + res1_F2;
    res2 = max(res2, res2_F2);
    res3(k * mn + (i - 1) * 4 * mn + 1 : ...
        k * mn + i * 4 * mn) = res3_F2;
    
end

% apply delta_{|| . ||_* <= nu} to y3
[res1_F3, res2_F3, res3_F3] = ...
    nuclear_norm_constraint(y3, k, sigma, nu, conjugate_flag);

% update outputs
res1 = res1 + res1_F3;
res2 = max(res2, res2_F3);
res3(5 * k * mn + 1 : end) = res3_F3;

end