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
% numFrames = 7;
% ex = 3;
% A = createTestImage(m, n, numFrames, ex);
% 
% k = 3;
% idx = ceil(numFrames / 2) - floor(k / 2) + (0 : k);
% img = cell(k + 1, 1);
% for i = 1 : k + 1
%     img{i} = A(:, :, idx(i));
% end
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-3;
% outerIter = 15;
% mu = 1e0;
% nu_factor = 0.8;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% % ~~~~~~~ ROTATING STAR DATA ~~~~~~~
% k = 1;
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
% k = 1;
% img{1} = double(rgb2gray(imread('sr1.png')));
% img{1} = normalize(img{1});
% img{2} = double(rgb2gray(imread('sr2.png')));
% img{2} = normalize(img{2});
% 
% % optimization parameters
% theta = 1;
% maxIter = 1000;
% tol = 1e-2;
% outerIter = 10;
% mu = 1e-1;
% nu_factor = 0.8;
% bc = 'linear';
% % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% ~~~~~~~~~ HEART MRI CINE ~~~~~~~~~
k = 5;
load('heart_mri.mat');
IDX = floor(linspace(beats(1, 1), beats(1, 2), k + 1));
factor = 4;
for i = 1 : (k + 1)
    % downsampling
    tmp = conv2(data(:, :, IDX(i)), ones(factor) / factor ^ 2, 'same');
    img{i} = tmp(1 : factor : end, 1 : factor : end);
end

% optimization parameters
theta = 1;
maxIter = 1000;
tol = 1e-2;
outerIter = 10;
mu = 2e-2;
nu_factor = 0.35;
bc = 'linear';
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
R = evaluate_displacement(img{k + 1}, h_img, zeros(m * n, 2), omega);

%% OPTIMIZATION SCHEME

% initialize primary and dual variables
x = zeros((3 * k + 1) * m * n, 1);
p = zeros((5 * k + 1) * m * n, 1);

% lower left block of A ~> gradient operator on displacements u
Dx = (1 / h_grid(1)) * spdiags([-ones(m, 1), ones(m, 1)], 0 : 1, m, m);
if strcmp(bc, 'linear')
    Dx(m, [m - 1, m]) = [-1, 1];
elseif strcmp(bc, 'neumann')
    Dx(m, m) = 0;
else
    error('Unknown boundary condition!');
end
Dy = (1 / h_grid(2)) * spdiags([-ones(n, 1), ones(n, 1)], 0 : 1, n, n);
if strcmp(bc, 'linear')
    Dy(n, [n - 1, n]) = [-1, 1];
elseif strcmp(bc, 'neumann')
    Dy(n, n) = 0;
else
    error('Unknown boundary condition!');
end
D = kron(speye(2), [kron(speye(n), Dx); kron(Dy, speye(m))]);
A2 = kron(speye(k), D);

% upper right block of A ~> identity matrix
A3 = speye((k + 1) * m * n);

% lower right block of A ~> all zeros
A4 = sparse(4 * k * m * n, (k + 1) * m * n);

% prepare figures for output display
figure(1);
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);
figure(2);
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);
green = cat(3, zeros(m, n), ones(m, n), zeros(m, n));

for o = 1 : outerIter
%-------------------------------------------------------------------------%
% BEGIN OUTER ITERATION
    
    % fetch u-part from primary variables
    u0 = x(1 : 2 * k * m * n);
    u0 = reshape(u0, m * n, 2, k);
    
    % reference vector for computing SAD from L
    b = zeros((k + 1) * m * n, 1);
    b(k * m * n + 1 : end) = vec(R);
    
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
    
    % estimate threshold nu from nuclear norm of column-wise images
    D = [reshape(T_current, m * n, k), vec(R)] - ...
                                            repmat(vec(R), [1, k + 1]);
    [~, S, ~] = svd(D, 'econ');
    nu = (nu_factor ^ (1 / outerIter)) * sum(diag(S));
    
    % upper left block of A ~> template image gradients
    A1 = [      -blkdiag(dT{:})
            sparse(m * n, 2 * k * m * n)    ];
    
    % build up A from the computed blocks
    A = [       A1,     A3
                A2,     A4          ];
    
    % estimate spectral norm of A
    norm_A_est = normest(A);
    
    % use estimated norm to get primal and dual step sizes
    tau = sqrt(0.95 / norm_A_est ^ 2);
    sigma = sqrt(0.95 / norm_A_est ^ 2);
    
    % get function handle to G-part of target function
    G_handle = @(x, c_flag) G(x, R, tau, nu, c_flag);
    
    % get function handle to F-part of target function
    F_handle = @(y, c_flag) F(y, b, k + 1, mu, sigma, c_flag);
    
    % perform optimization
    [x, p, primal_history, dual_history] = chambolle_pock(...
        F_handle, G_handle, A, x, p, theta, tau, sigma, maxIter, tol);
    
    %% ITERATIVE OUTPUT
    
    % plot primal & dual energies
    figure(1);
    clf;
    
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
    L_star = reshape(L_star, m, n, k + 1);
    
    % get cell-centered grid over omega for plotting purposes
    [cc_x, cc_y] = cell_centered_grid(omega, [m, n]);
    cc_grid = [cc_x(:), cc_y(:)];
    
    % display resulting displacements & low rank components
    figure(2);
    clf;
    colormap gray(256);
    
    for i = 1 : k
        subplot(3, k + 1, i);
        set(gca, 'YDir', 'reverse');
        imagesc(...
            'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
            'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
            'CData', T(:, :, i));
        axis image;
        title(sprintf('template T_%d', i));
        hold on;
        quiver(cc_grid(:, 2), cc_grid(:, 1), ...
            u_star(:, 2, i), u_star(:, 1, i), 0, 'r');
%         plot_grid(reshape(cc_grid + u_star(:, :, i), [m, n, 2]), 2);
        hold off;
    end
    
    subplot(3, k + 1, k + 1);
    set(gca, 'YDir', 'reverse');
    imagesc(...
        'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
        'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
        'CData', R);
    axis image;
    title('reference R');
    
    for i = 1 : k
        subplot(3, k + 1, (k + 1) + i);
        set(gca, 'YDir', 'reverse');
        imagesc(...
            'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
            'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
            'CData', T_u(:, :, i));
        axis image;
        title(sprintf('T_%d(u_%d)', i, i));
        hold on;
        imagesc(...
            'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
            'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
            'CData', green, ...
            'AlphaData', abs(R - T_u(:, :, i)));
        hold off;
    end
    
    for i = 1 : (k + 1)
        subplot(3, k + 1, 2 * (k + 1) + i);
        set(gca, 'YDir', 'reverse');
        imagesc(...
            'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
            'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
            'CData', L_star(:, :, i) - R);
        axis image;
        colorbar;
        title(sprintf('low rank component l_%d - R', i));
    end
    
    % pause until button press
    if o < outerIter
        str = [sprintf('%s ITERATION #%d FINISHED %s\n\n', ...
            repmat('~', [1, 10]), o, repmat('~', [1, 10])), ...
            'PRESS ENTER TO CONTINUE: '];
        input(str);
    end
    
% END OUTER ITERATION
%-------------------------------------------------------------------------%
end

%% LOCAL FUNCTION DEFINITIONS

function [res1, res2, res3] = G(x, R, tau, nu, conjugate_flag)
% extends nuclear_norm_constraint.m to work on inputs x = [u; l]
%   with u <-> displacement fields and l <-> low rank images

% get number of template images and number of pixels per image
mn = numel(R);
k = (numel(x) / mn - 1) / 3;

% split input x into u- and l-part
u = x(1 : 2 * k * mn);
l = x(2 * k * mn + 1 : end);

% apply nuclear norm constraint on l-part (compute only if requested!)
if nargout == 3
    
    [res1, res2, res3] = ...
        nuclear_norm_constraint_mod(l, R, tau, nu, conjugate_flag);
    
    if ~conjugate_flag
        % prox for 0 * u is identity
        res3 = [u; res3];
    else
        % prox for delta_{0}(u) = 0
        res2 = max(res2, max(abs(u(:))));
        res3 = [zeros(size(u)); res3];
    end
    
else
    
    [res1, res2] = ...
        nuclear_norm_constraint_mod(l, R, tau, nu, conjugate_flag);
    
    if conjugate_flag
        res2 = max(res2, max(abs(u(:))));
    end
    
end

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

% apply SAD to r-part (compute prox only if requested!)
if nargout == 3
    
    [res1_F1, res2_F1, res3_F1] = SAD(r, b, sigma, conjugate_flag);
    
    % initialize outputs with values from F1
    res1 = res1_F1;
    res2 = res2_F1;
    res3 = zeros((5 * k + 1) * mn, 1);
    res3(1 : (k + 1) * mn) = res3_F1;
    
    % apply mu*||.||_{2,1} to each of the k components v_i of v
    v = reshape(v, 4 * mn, k);
    for i = 1 : k
        
        [res1_F2, res2_F2, res3_F2] = ...
            norm21(v(:, i), mu, sigma, conjugate_flag);
        
        % update outputs
        res1 = res1 + res1_F2;
        res2 = max(res2, res2_F2);
        res3((k + 1) * mn + (i - 1) * 4 * mn + 1 : ...
            (k + 1) * mn + i * 4 * mn) = res3_F2;
        
    end
    
else
    
    [res1_F1, res2_F1] = SAD(r, b, sigma, conjugate_flag);
    
    % initialize outputs with values from F1
    res1 = res1_F1;
    res2 = res2_F1;
    
    % apply mu*||.||_{2,1} to each of the k components v_i of v
    v = reshape(v, 4 * mn, k);
    for i = 1 : k
        
        [res1_F2, res2_F2] = norm21(v(:, i), mu, sigma, conjugate_flag);
        
        % update outputs
        res1 = res1 + res1_F2;
        res2 = max(res2, res2_F2);
        
    end
    
end

end