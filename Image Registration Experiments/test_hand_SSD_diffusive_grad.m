%% initialization
clear all, close all, clc;

R = double(imread('hand1.png'));
T = double(imread('hand2.png'));
[m, n] = size(R);
h = [1, 1];

% display reference and template image
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(1, 2, 1);
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(R));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
title('reference R');

subplot(1, 2, 2);
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(T));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
title('template T');

%% registration procedure

% set function handles for data term, regularizer and final objective
dist_fctn = @(T, R, h, u) SSD(T, R, h, u);
reg_fctn = @(u, s, h) diffusive_energy(u, s, h);
lambda = 1e4;
f = @(u) objective_function(dist_fctn, reg_fctn, lambda, T, R, h, u);

% set optimization parameters
u0 = zeros(m * n * 2, 1);
tol1 = 1e-1;
maxIter = 500;
tol2 = 1e-3;

% perform optimization and evaluate result
u_star = gradient_descent(f, u0, tol1, maxIter, tol2);
u_star = reshape(u_star, [m*n, 2]);
T_u_star = evaluate_displacement(T, h, u_star);

% compute grid g from displacement u
[cc_x, cc_y] = cell_centered_grid([m, n], h);
g = [cc_x(:), cc_y(:)] + u_star;
g = reshape(g, [m, n, 2]);

%% display results
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(1, 2, 1);
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(T));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
plot_grid(g);
title('template T with displaced grid')

subplot(1, 2, 2);
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(T_u_star));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
title('transformed template T_u');