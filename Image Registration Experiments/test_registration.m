%% choose data / regularizer / optimization scheme
clear all, close all, clc;

% 1. choose data from {'rect', 'hand'}
data = 'hand';

% 2. choose regularizer from {'diffusive', 'curvature'}
regularizer = 'curvature';

% 3. choose optimizer from {'gradient_descent', 'newton'}
optimizer = 'newton';

%% initialization

if strcmp(data, 'rect')
    R = double(imread('rect1.png'));
    T = double(imread('rect2.png'));
elseif strcmp(data, 'hand')
    check_hand_data;
    R = double(imread('hands-R.jpg'));
    T = double(imread('hands-T.jpg'));
end

[m, n] = size(R);
h = [1, 1];

% display reference and template image
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(2, 2, 1);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', R);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('reference R');

subplot(2, 2, 2);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('template T');

%% registration procedure

% set function handles for data term, regularizer and final objective
dist_fctn = @(T, R, h, u) SSD(T, R, h, u);

if strcmp(regularizer, 'curvature')
    reg_fctn = @(u, s, h) curvature_energy(u, s, h);
elseif strcmp(regularizer, 'diffusive')
    reg_fctn = @(u, s, h) diffusive_energy(u, s, h);
end

lambda = 5e5;
f = @(u) objective_function(dist_fctn, reg_fctn, lambda, T, R, h, u);

% optimization procedure
u0 = zeros(m * n * 2, 1);
if strcmp(optimizer, 'gradient_descent')
    u_star = gradient_descent(f, u0);
elseif strcmp(optimizer, 'newton')
    u_star = newton_scheme(f, u0);
end

% evaluate result
u_star = reshape(u_star, [m*n, 2]);
T_u_star = evaluate_displacement(T, h, u_star);

% compute grid g from displacement u
[cc_x, cc_y] = cell_centered_grid([m, n], h);
g = [cc_x(:), cc_y(:)] + u_star;
g = reshape(g, [m, n, 2]);

%% display results

subplot(2, 2, 3);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
plot_grid(g, 4);
title('template T with displaced grid')

subplot(2, 2, 4);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T_u_star);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('transformed template T_u');