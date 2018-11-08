%% initialization
clear all, close all, clc;

R = double(imread('hand1.png'));
T = double(imread('hand2.png'));
[m, n] = size(R);
h = [1, 1];

% create multi-level versions of R and T
R_ML = multi_level(R, h);
T_ML = multi_level(T, h);
num_levels = length(R_ML);

%% set up registration at lowest resolution
dist_fctn = @(T, R, h, u) SSD(T, R, h, u);
reg_fctn = @(u, s, h) curvature_energy(u, s, h);
lambda = 5e4;
f = @(u) objective_function(dist_fctn, reg_fctn, lambda, ...
    T_ML{1}.X, R_ML{1}.X, R_ML{1}.h, u);

% set optimization parameters
u0 = zeros(prod(R_ML{1}.s) * 2, 1);
tol1 = 1e-1;
maxIter = 2000;
tol2 = 1e-3;

% some output
str = sprintf('REGISTRATING AT LEVEL %d WITH RESOLUTION [%d, %d]', ...
    1, R_ML{1}.s(1), R_ML{1}.s(2));
fprintf('\n%s%s%s\n', ...
    repmat('~', [1, floor(0.5 * (80 - length(str)))]), ...
    str, ...
    repmat('~', [1, ceil(0.5 * (80 - length(str)))]));

% perform optimization
u_star = gradient_descent(f, u0, tol1, maxIter, tol2);

%% iterate up to input resolution
for i = 2 : num_levels
    
    % use prolonged minimizer of low res as starting solution for high res
    u_prolonged = prolong_displacement(...
        u_star, T_ML{i - 1}.s, T_ML{i}.s);
    
    % update objective function to high res
    f = @(u) objective_function(dist_fctn, reg_fctn, lambda, ...
        T_ML{i}.X, R_ML{i}.X, R_ML{i}.h, u);
    
    % more output
    str = sprintf('REGISTRATING AT LEVEL %d WITH RESOLUTION [%d, %d]', ...
        i, R_ML{i}.s(1), R_ML{i}.s(2));
    fprintf('\n%s%s%s\n', ...
        repmat('~', [1, floor(0.5 * (80 - length(str)))]), ...
        str, ...
        repmat('~', [1, ceil(0.5 * (80 - length(str)))]));
    
    u_star = gradient_descent(f, u_prolonged, tol1, maxIter, tol2);
end

% evaluate output
u_star = reshape(u_star, [m * n, 2]);
T_u_star = evaluate_displacement(...
    T_ML{num_levels}.X, T_ML{num_levels}.h, u_star);

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