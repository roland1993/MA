%   min_u delta_{|| . || <= nu}(B * [I_1(u_1), .., I_k(u_k)])
%           + mu * sum_i TV(u_i) + || u_mean ||_2^2
%
%   NO LOW-RANK & MEAN-FREE & NO REFERENCE & USES UNIQUENESS-TERM

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
% k = 4;
% A = createTestImage(32, 32, 9, 3);
% idx = ceil(size(A, 3) / 2) - floor(k / 2) + (1 : k);
% img = cell(k, 1);
% for i = 1 : k, img{i} = normalize(A(:, :, idx(i))); end
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 0;
% outerIter = 5;
% mu = 1e0;
% nu_factor = 0.8;
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
% tol = 0;
% outerIter = 10;
% mu = 1e-1;
% nu_factor = 0.75;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% % ~~~~~~~ SLIDING RECT DATA ~~~~~~~
% k = 2;
% img{1} = normalize(double(rgb2gray(imread('sr1.png'))));
% img{2} = normalize(double(rgb2gray(imread('sr2.png'))));
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 0;
% outerIter = 7;
% mu = 3e-1;
% nu_factor = 0.65;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% ~~~~~~~~~ DYNAMIC TEST IMAGES ~~~~~~~~~
m = 100;
n = 100;
k = 8;
data = dynamicTestImage(m, n, k);
img = cell(1, k);
for i = 1 : k, img{i} = data(:, :, i);  end

% optimization parameters
theta = 1;
maxIter = 2000;
tol = 1e-3;
outerIter = 20;
mu = 2e-1;
nu_factor = 0.9;
bc = 'linear';
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% % ~~~~~~~~~ HEART MRI CINE ~~~~~~~~~
% k = 4;
% load('heart_mri.mat');
% IDX = floor(linspace(beats(1, 1), beats(1, 2), k));
% factor = 4;
% for i = 1 : k
%     % downsampling
%     tmp = conv2(data(:, :, IDX(i)), ones(factor) / factor ^ 2, 'same');
%     img{i} = tmp(1 : factor : end, 1 : factor : end);
% end
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-2;
% outerIter = 10;
% mu = 2e-2;
% nu_factor = 0.4;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% get image resolution etc.
[m, n] = size(img{1});
h_img = [1, 1];
omega = [0, m, 0, n];
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% evaluate template images over chosen omega-region
T = zeros(m, n, k);
for i = 1 : k
    T(:, :, i) = evaluate_displacement( ...
        img{i}, h_img, zeros(m * n, 2), omega);
end

%% OPTIMIZATION SCHEME

% initialize primary and dual variables
x = zeros(2 * k * m * n, 1);
p = zeros(5 * k * m * n + 2, 1);

% get mean free operator
M = mean_free_operator(m, n, k);

% middle block of A ~> gradient operator on displacements u
A2 = finite_difference_operator(m, n, h_grid, k, bc);

% lower block of A ~> mean of u in x- and y-direction
A3 = 1 / (m * n * k) * ...
    [   kron(ones(1, k), [ones(1, m * n), sparse(1, m * n)])
        kron(ones(1, k), [sparse(1, m * n), ones(1, m * n)])    ];

% prepare figures for output display
fh1 = figure(1);
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);
fh2 = figure(2);
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);
green = cat(3, zeros(m, n), ones(m, n), zeros(m, n));

% get cell-centered grid over omega for plotting purposes
[cc_x, cc_y] = cell_centered_grid(omega, [m, n]);
cc_grid = [cc_x(:), cc_y(:)];

% track change in singluar values
SV_history = zeros(k, outerIter);

for o = 1 : outerIter
%-------------------------------------------------------------------------%
% BEGIN OUTER ITERATION
    
    % fetch u-part from primary variables
    u0 = reshape(x, m * n, 2, k);
    
    % constant offset vector for nuclear norm
    b = zeros(k * m * n, 1);
    
    % templates evaluated using current u
    T_current = zeros(m, n, k);
    
    % store image gradients of template images, fill up vector b
    dT = cell(k, 1);
    for i = 1 : k
        [T_current(:, :, i), dT{i}] = ...
            evaluate_displacement(img{i}, h_img, u0(:, :, i), omega);
        b((i - 1) * m * n + 1 : i * m * n) = ...
            vec(T_current(:, :, i)) - dT{i} * vec(u0(:, :, i));
    end
    
    % upper block of A ~> template image gradients
    A1 = M * blkdiag(dT{:});
    
    % build up A from the computed blocks
    A = [   	A1
                A2      
                A3      ];
    
    % estimate spectral norm of A
    norm_A_est = normest(A);
    
    % use estimated norm to get primal and dual step sizes
    tau = sqrt(0.975 / norm_A_est ^ 2);
    sigma = sqrt(0.975 / norm_A_est ^ 2);
    
    % get function handle to G-part of target function
    G_handle = @(x, c_flag) zero_function(x, c_flag);
    
    % estimate nu-parameter
    I = reshape(M * T_current(:), m * n, k);
    [~, SV, ~] = svd(I, 'econ');
    nu = nu_factor * sum(diag(SV));
    
    % get function handle to F-part of target function
    b = M * (-b);
    F_handle = @(y, c_flag) F(y, b, k, nu, mu, sigma, c_flag);
    
    % perform optimization
    [x, p, primal_history, dual_history] = chambolle_pock(...
        F_handle, G_handle, A, x, p, theta, tau, sigma, maxIter, tol);
    
    % ITERATIVE OUTPUT
    
    % plot primal & dual energies
    figure(fh1);
    clf;
    set(fh1, 'NumberTitle', 'off', ...
        'Name', sprintf('PLOTS - ITERATE %d OUT OF %d', o, outerIter));
    
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
    
    % evaluate minimizer u_star
    u_star = reshape(x, m * n, 2, k);
    T_u = zeros(m, n, k);
    for i = 1 : k
        T_u(:, :, i) = ...
            evaluate_displacement(img{i}, h_img, u_star(:, :, i), omega);
    end
    meanImg = sum(T_u, 3) / k;
    [~, SV, ~] = svd(reshape(T_u - meanImg, [], k), 'econ');
    SV_history(:, o) = vec(diag(SV));
    
    % display resulting displacements
    figure(fh2);
    clf;
    colormap gray(256);
    set(fh2, 'NumberTitle', 'off', ...
        'Name', sprintf('RESULTS - ITERATE %d OUT OF %d', o, outerIter));
    
    for i = 1 : k
        subplot(2, k, i);
        imshow(T(:, :, i), [0, 1], 'InitialMagnification', 'fit');
        title(sprintf('T_%d', i));
        hold on;
        quiver(cc_grid(:, 2), cc_grid(:, 1), ...
            u_star(:, 2, i), u_star(:, 1, i), 0, 'r');
%         plot_grid(reshape(cc_grid + u_star(:, :, i), [m, n, 2]), 2);
        hold off;
        axis image;
    end
    
    for i = 1 : k
        subplot(2, k, k + i);
        imshow(T_u(:, :, i), [0, 1], 'InitialMagnification', 'fit');
        title(sprintf('T_%d(u_%d)', i, i));
        hold on;
        imagesc(...
            'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
            'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
            'CData', green, ...
            'AlphaData', abs(T_u(:, :, i) - meanImg));
        hold off;
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
title('singular values of mean-free images');
grid on;
legend(names);

%% LOCAL FUNCTION DEFINITIONS

function [res1, res2, res3] = ...
    F(y, d, k, nu, mu, sigma, conjugate_flag)
% splits input y = [v; w] and computes
%   F_1(v) = delta_{|| . ||_* <= nu}(y - d)
%   F_2(w) = sum_i mu * || w_i ||_{2,1}
%   F_3(x) = || x ||_2^2

% get number of pixels per image
mn = (numel(y) - 2) / (5 * k);

% split input y into v- and w-part
v = y(1 : (k * mn));
w = y((k * mn + 1) : (5 * k * mn));
x = y((5 * k * mn + 1) : end);

% for the sake of efficiency: compute prox only if requested!
if nargout == 3
    
    % apply nuclear norm to v-part
    [res1_F1, res2_F1, res3_F1] = nuclear_norm_constraint_mod( ...
        v, d, k, sigma, nu, conjugate_flag);
    
    % initialize outputs with values from F1
    res1 = res1_F1;
    res2 = res2_F1;
    res3 = zeros(5 * k * mn + 2, 1);
    res3(1 : (k * mn)) = res3_F1;
    
    % apply mu * ||.||_{2,1} to each of the k components v_i of v
    w = reshape(w, 4 * mn, k);
    for i = 1 : k
        [res1_F2, res2_F2, res3_F2] = ...
            norm21(w(:, i), mu, sigma, conjugate_flag);
        % update outputs
        res1 = res1 + res1_F2;
        res2 = max(res2, res2_F2);
        res3(k * mn + (i - 1) * 4 * mn + 1 : ...
            k * mn + i * 4 * mn) = res3_F2;
    end
    
    % apply || . ||_2^2 to x
    % [~, ~, res3_F3] = norm2_squared(x, sigma, conjugate_flag);
    
    % apply delta_{0}(.) to x
    [res1_F3, res2_F3, res3_F3] = zero_function(x, ~conjugate_flag);
    % update outputs
    res1 = res1 + res1_F3;
    res2 = max([res2, res2_F3]);
    res3((5 * k * mn + 1) : end) = res3_F3;
    
else
    
    % apply nuclear norm to v-part
    [res1_F1, res2_F1] = nuclear_norm_constraint_mod( ...
        v, d, k, sigma, nu, conjugate_flag);
    
    % initialize outputs with values from F1
    res1 = res1_F1;
    res2 = res2_F1;
    
    % apply mu * ||.||_{2,1} to each of the k components v_i of v
    w = reshape(w, 4 * mn, k);
    for i = 1 : k
        [res1_F2, res2_F2] = ...
            norm21(w(:, i), mu, sigma, conjugate_flag);
        % update outputs
        res1 = res1 + res1_F2;
        res2 = max(res2, res2_F2);
    end
    
    % apply || . ||_2^2 to x
    % [res1_F3, res2_F3] = norm2_squared(x, sigma, conjugate_flag);
    
    % apply delta_{0}(.) to x
    [res1_F3, res2_F3] = zero_function(x, ~conjugate_flag);
    % update outputs
    res1 = res1 + res1_F3;
    res2 = max([res2, res2_F3]);
    
end

end