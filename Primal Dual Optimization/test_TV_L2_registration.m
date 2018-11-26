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

% read reference and template images
R = double(imread('rect3.png'));
T = double(imread('rect4.png'));

% normalize images
R = (R - min(R(:))) / (max(R(:)) - min(R(:)));
T = (T - min(T(:))) / (max(T(:)) - min(T(:)));

% image & grid parameters
[m, n] = size(R);
h = [1, 1];

%% setup and optimization

lambda = 0.01;

% define discrete gradient operator K
Dx = (1 / h(1)) * spdiags([-ones(m, 1), ones(m, 1)], 0 : 1, m, m);
Dx(m, m) = 0;
Dy = (1 / h(2)) * spdiags([-ones(n, 1), ones(n, 1)], 0 : 1, n, n);
Dy(n, n) = 0;
Gx = kron(speye(n), Dx);    Gy = kron(Dy, speye(m));
K = lambda * kron(speye(2), [Gx; Gy]);

% upper bound on spectral norm of K
L_squared = 4 * lambda ^ 2 * (1 / h(1) ^ 2 + 1 / h(2) ^ 2);

% set parameters of optimization scheme 
u0 = zeros(m * n * 2, 1);
v0 = zeros(m * n * 4, 1);
theta = 1;
tau = 100;
sigma = 1 / (L_squared * tau);
maxIter = 30;

% function handles for data term and regularizer
G = @(u, c_flag) SSD_registration(u, u0, T, R, h, tau, c_flag);
F = @(v, c_flag) TV_registration(v, sigma, c_flag);

figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

% perform optimization
for i = 1 : 40
    
    T_star = evaluate_displacement(T, h, reshape(u0, [], 2));
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', T_star);
    axis image;     set(gca, 'YDir', 'reverse');
    drawnow;
    title(sprintf('i = %d', i));
    
    [u_star, v_star] = chambolle_pock(F, G, K, u0, v0, theta, tau, ...
        sigma, maxIter, -1, -1, -1);
    
    u0 = u_star;    v0 = v_star;
    G = @(u, c_flag) SSD_registration(u, u_star, T, R, h, tau, c_flag);
end

%% display results

figure('units', 'normalized', 'outerposition', [0 0 1 1]);
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
title('template image T');

subplot(2, 2, 3);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('template image T with displaced grid');
[xx, yy] = cell_centered_grid([m, n], h);
g = cat(3, xx, yy) + reshape(u_star, m, n, 2);
plot_grid(g, 2);

T_star = evaluate_displacement(T, h, reshape(u0, m * n, 2));
subplot(2, 2, 4);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T_star);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('transformed template T\_star');