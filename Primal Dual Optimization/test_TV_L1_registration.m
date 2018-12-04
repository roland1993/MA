%% initialization

% clean-up
clear all, close all, clc;

% make sure to have interpolation routines available
if exist('evaluate_displacement.m') == 0
    pwd_str = pwd;
    tmp = strfind(pwd_str, '/');
    % .. should be found from parent directory
    parent_directory = pwd_str(1 : tmp(end));
    addpath(genpath(parent_directory));
end

% choose data from {'rect', 'rect_in_rect', 'sliding_rect'}
data = 'rect_in_rect';
switch data
    
    case 'rect'
        R = double(imread('rect_1.png'));
        T = double(imread('rect_2.png'));
        lambda = 0.7;
        tau = 5;
        maxIter = 30;
        numSteps = 40;
        
    case 'rect_in_rect'
        R = double(imread('rect_in_rect_1.png'));
        T = double(imread('rect_in_rect_2.png'));
        lambda = 10;
        tau = 5;
        maxIter = 25;
        numSteps = 40;
        
    case 'sliding_rect'
        R = double(imread('sliding_rect_1.png'));
        T = double(imread('sliding_rect_2.png'));
        lambda = 4;
        tau = 1.5;
        maxIter = 30;
        numSteps = 40;
        
end

% normalize images
R = (R - min(R(:))) / (max(R(:)) - min(R(:)));
T = (T - min(T(:))) / (max(T(:)) - min(T(:)));

% image & grid parameters
[m, n] = size(R);
h = [1, 1];

%% setup and optimization

% define discrete gradient operator K
Dx = (1 / h(1)) * spdiags([-ones(m, 1), ones(m, 1)], 0 : 1, m, m);
Dx(m, m) = 0;
Dy = (1 / h(2)) * spdiags([-ones(n, 1), ones(n, 1)], 0 : 1, n, n);
Dy(n, n) = 0;
Gx = kron(speye(n), Dx);    Gy = kron(Dy, speye(m));
K = kron(speye(2), [Gx; Gy]);

% upper bound on spectral norm of K
L_squared = 4 * (1 / h(1) ^ 2 + 1 / h(2) ^ 2);

% set parameters of optimization scheme
u0 = zeros(m * n * 2, 1);
v0 = zeros(m * n * 4, 1);
theta = 1;
sigma = (1 - 1e-4) / (L_squared * tau);

% function handles for data term and regularizer
G = @(u, c_flag) SAD_registration(u, u0, T, R, h, lambda, tau, c_flag);
F = @(v, c_flag) TV_registration(v, sigma, c_flag);

figure('units', 'normalized', 'outerposition', [0 0 0.5 1]);
colormap gray(256);
[xx, yy] = cell_centered_grid([m, n], h);

% perform optimization
for i = 1 : numSteps
    
    clf;
    subplot(1, 2, 1);
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', T);
    axis image;     set(gca, 'YDir', 'reverse');
    g = cat(3, xx, yy) + reshape(u0, m, n, 2);
    plot_grid(g);
    title(sprintf('i = %d', i));
    
    subplot(1, 2, 2);
    T_star = evaluate_displacement(T, h, reshape(u0, [], 2));
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', T_star);
    axis image;     set(gca, 'YDir', 'reverse');
    drawnow;
    
    [u_star, v_star] = ...
        chambolle_pock(F, G, K, u0, v0, theta, tau, sigma, maxIter);
    
    u0 = u_star;    v0 = v_star;
    G = @(u, c_flag) SAD_registration(u, u0, T, R, h, lambda, tau, c_flag);
    
end

%% display results

figure('units', 'normalized', 'outerposition', [0.5 0 0.5 1]);
colormap gray(256);

subplot(2, 2, 1);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', R);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('reference image R');

subplot(2, 2, 2);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('template image T with displaced grid');
g = cat(3, xx, yy) + reshape(u_star, m, n, 2);
plot_grid(g, 2);

T_star = evaluate_displacement(T, h, reshape(u0, m * n, 2));
subplot(2, 2, 3);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T_star);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('transformed template T\_star');

subplot(2, 2, 4);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', abs(T_star - R));
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('absolute difference |T\_star - R|');