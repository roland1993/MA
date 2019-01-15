%% INITIALIZATION

% clean-up
clear all, clc, close all;

% make sure that interpolation routines are on search path
if ~exist('evaluate_displacement.m', 'file')
    addpath(genpath('..'));
end

% load respiratory data if available
if ~exist('respfilm1gray.mat', 'file')
    error('Missing file: ''respfilm1gray.mat''!');
end


% load film and normalize to [0 1]
load('respfilm1gray.mat', 'A');
A = (A - min(A(:))) / (max(A(:)) - min(A(:)));

% k = number of template frames
k = 2;
idx = floor(linspace(1, size(A, 3), k + 1));

% downsample frames
img = cell(k + 1, 1);
for i = 1 : k + 1
    img{i} = conv2(A(:, :, idx(i)), ones(4) / 16, 'same');
    img{i} = img{i}(1 : 4 : end, 1 : 4 : end);
end

% get image resolution etc.
[m, n] = size(img{1});
h_img = 1 ./ size(img{1});
omega = [-0.25, 1.25, -0.25, 1.25];
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% choose reference image
R = evaluate_displacement(img{k + 1}, h_img, zeros(m * n, 2), omega);
img(k + 1) = [];

% column major vectorization operator
vec = @(x) x(:);

%% OPTIMIZATION SCHEME

% development point for 1st order approximation of image model
u0 = zeros(m * n, 2, k);

% reference vector for computing SAD from L
b = zeros((k + 1) * m * n, 1);
b(k * m * n + 1 : end) = vec(R);

% store image gradients of template images, complete vector b
dT = cell(k, 1);
for i = 1 : k
    [T, dT{i}] = ...
        evaluate_displacement(img{i}, h_img, u0(:, :, i), omega);
    b((i - 1) * m * n + 1 : i * m * n) = ...
        vec(T) - dT{i} * vec(u0(:, :, i));
end

% estimate threshold parameter nu from nuclear norm of 
%   matrix of column-wise images (constructed from vector b)
[~, S, ~] = svd(reshape(b, m * n, k + 1), 'econ');
nu = 0.85 * sum(diag(S));

% upper left block of A ~> template image gradients
A1 = blkdiag(dT{:});
[i, j, v] = find(A1);
A1 = sparse(i, j, v, (k + 1) * m * n, 2 * k * m * n);
% A1 = sparse((k + 1) * m * n, 2 * k * m * n);

% lower left block of A ~> gradient operator for grids u
Dx = (1 / h_grid(1)) * spdiags([-ones(m, 1), ones(m, 1)], 0 : 1, m, m);
Dx(m, m) = 0;
Dy = (1 / h_grid(2)) * spdiags([-ones(n, 1), ones(n, 1)], 0 : 1, n, n);
Dy(n, n) = 0;
D = kron(speye(2), [kron(speye(n), Dx); kron(Dy, speye(m))]);
A2 = kron(speye(k), D);
% A2 = sparse(4 * k * m * n, 2 * k * m * n);

% upper right block of A ~> identity matrix
A3 = speye((k + 1) * m * n);

% lower right block of A ~> all zeros
A4 = sparse(4 * k * m * n, (k + 1) * m * n);

% build up A from the computed blocks
A = [       A1,     A3
            A2,     A4          ];

% estimate (squared) spectral norm of A via simple upper bound
% norm_A_squared = max(sum(abs(A), 1)) * max(sum(abs(A), 2));
norm_A_est = normest(A);

% use estimated norm to get primal and dual step sizes
% tau = sqrt((1 - 1e-4) / norm_A_squared);
tau = sqrt(0.95 / norm_A_est ^ 2);
sigma = sqrt(0.95 / norm_A_est ^ 2);

% get function handle to G-part of target function
G_handle = @(x, c_flag) G(x, k + 1, tau, nu, c_flag);

% regularization strength
mu = 1;

% get function handle to F-part of target function
F_handle =  @(y, c_flag) F(y, b, k + 1, mu, sigma, c_flag);

% miscellaneous optimization parameters
theta = 1;
x0 = zeros((3 * k + 1) * m * n, 1);
p0 = zeros((5 * k + 1) * m * n, 1);
maxIter = 10000;
tol = 1e0;

% perform optimization
[x_star, y_star, primal_history, dual_history] = ...
    chambolle_pock(F_handle, G_handle, A, x0, p0, ...
    theta, tau, sigma, maxIter, tol);

%% SOME OUTPUT

% plot primal & dual energies
figure('units', 'normalized', 'outerposition', [0 0.5 1 0.5])

subplot(1, 3, 1);
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

subplot(1, 3, 2);
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

subplot(1, 3, 3);
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

% evaluate minimizer x_star = [u_star; L_star]
u_star = x_star(1 : 2 * k * m * n);
u_star = reshape(u_star, m * n, 2, k);
L_star = x_star(2 * k * m * n + 1 : end);
L_star = reshape(L_star, m, n, k + 1);

% get cell-centered grid over omega for plotting purposes
[cc_x, cc_y] = cell_centered_grid(omega, [m, n]);
cc_grid = [cc_x(:), cc_y(:)];

% only display the longest 50% of displacement vectors
u_star_length = squeeze(sqrt(sum(u_star .^ 2, 2)));
q1 = quantile(u_star_length(:, 1), 0.5);
idx1 = (u_star_length(:, 1) >= q1);
q2 = quantile(u_star_length(:, 2), 0.5);
idx2 = (u_star_length(:, 2) >= q2);

% display resulting displacements & low rank components
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(3, 3, 1);
set(gca, 'YDir', 'reverse');
imagesc(...
    'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
    'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
    'CData', evaluate_displacement(img{1}, h_img, zeros(m * n, 2), omega));
axis image;
title('template T_1');

hold on;
quiver(cc_grid(idx1, 2), cc_grid(idx1, 1), ...
    u_star(idx1, 2, 1), u_star(idx1, 1, 1), 0, 'b');
hold off;

subplot(3, 3, 2);
set(gca, 'YDir', 'reverse');
imagesc(...
    'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
    'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
    'CData', evaluate_displacement(img{2}, h_img, zeros(m * n, 2), omega));
axis image;
title('template T_2');

hold on;
quiver(cc_grid(idx2, 2), cc_grid(idx2, 1), ...
    u_star(idx2, 2, 2), u_star(idx2, 1, 2), 0, 'b');
hold off;

subplot(3, 3, 3);
set(gca, 'YDir', 'reverse');
imagesc(...
    'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
    'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
    'CData', R);
axis image;
title('reference R');

subplot(3, 3, 4);
set(gca, 'YDir', 'reverse');
imagesc(...
    'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
    'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
    'CData', evaluate_displacement(img{1}, h_img, u_star(:, :, 1), omega));
axis image;
title('T_1(u_1)');

subplot(3, 3, 5);
set(gca, 'YDir', 'reverse');
imagesc(...
    'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
    'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
    'CData', evaluate_displacement(img{2}, h_img, u_star(:, :, 2), omega));
axis image;
title('T_2(u_2)');

for i = 1 : 3
    subplot(3, 3, 6 + i);
    set(gca, 'YDir', 'reverse');
    imagesc(...
        'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
        'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
        'CData', L_star(:, :, i));
    axis image;
    title(sprintf('low rank component l_%d', i));
end

%% LOCAL FUNCTION DEFINITIONS

function [res1, res2, res3] = G(x, numImg, tau, nu, conjugate_flag)
% extends nuclear_norm_constraint.m to work on inputs x = [u; l]
%   with u <-> displacement fields and l <-> low rank images

% get number of template images and number of pixels per image
k = numImg - 1;
mn = numel(x) / (3 * k + 1);

% split input x into u- and l-part
u = x(1 : 2 * k * mn);
l = x(2 * k * mn + 1 : end);

% apply nuclear norm constraint only on l-part
[res1, res2, res3] = ...
    nuclear_norm_constraint(l, numImg, tau, nu, conjugate_flag);

% extend res2 = prox(l) to a proper prox of same length as x
res2 = [u; res2];

end

function [res1, res2, res3] = F(y, b, numImg, mu, sigma, conjugate_flag)
% splits input y = [r; v] and computes
%   F_1(r) = ||r - b||_1, F_2(v) = sum_i mu * ||v_i||_{2,1}

% get number of template images and number of pixels per image
k = numImg - 1;
mn = numel(y) / (5 * k + 1);

% split input y into r- and v-part
r = y(1 : (k + 1) * mn);
v = y((k + 1) * mn + 1 : end);

% apply SAD to r-part
[res1_F1, res2_F1, res3_F1] = SAD(r, b, sigma, conjugate_flag);

% initialize outputs with values from F1
res1 = res1_F1;
res2 = zeros((5 * k + 1) * mn, 1);
res2(1 : (k + 1) * mn) = res2_F1;
res3 = res3_F1;

% apply mu*||.||_{2,1} to each of the k components v_i of v
v = reshape(v, 4 * mn, k);
for i = 1 : k
    
    [res1_F2, res2_F2, res3_F2] = ...
        norm21(v(:, i), mu, sigma, conjugate_flag);
    
    % update outputs
    res1 = res1 + res1_F2;
    res2((k + 1) * mn + (i - 1) * 4 * mn + 1 : ...
                                (k + 1) * mn + i * 4 * mn) = res2_F2;
    res3 = max(res3, res3_F2);
    
end

end