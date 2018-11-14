%% initialization
clear all, close all, clc;

check_hand_data;
R = double(imread('hands-R.jpg'));
T = double(imread('hands-T.jpg'));
[m, n] = size(R);
h = [1, 1];

% create multi-level versions of R and T
R_ML = multi_level(R, h);
T_ML = multi_level(T, h);
num_levels = length(R_ML);

%% set up registration at lowest resolution
dist_fctn = @(T, R, h, u) SSD(T, R, h, u);
reg_fctn = @(u, s, h) curvature_energy(u, s, h);
lambda = 5e5;
f = @(u) objective_function(dist_fctn, reg_fctn, lambda, ...
    T_ML{1}.img, R_ML{1}.img, R_ML{1}.h, u);

% set optimization parameters
u0 = zeros(prod(R_ML{1}.s) * 2, 1);

% some output
str = sprintf('REGISTRATING AT LEVEL %d WITH RESOLUTION [%d, %d]', ...
    1, R_ML{1}.s(1), R_ML{1}.s(2));
fprintf('\n%s%s%s\n', ...
    repmat('~', [1, floor(0.5 * (80 - length(str)))]), ...
    str, ...
    repmat('~', [1, ceil(0.5 * (80 - length(str)))]));

% perform optimization
u_star = newton_scheme(f, u0);

%% iterate up to input resolution
for i = 2 : num_levels
    
    % use prolonged minimizer of low res as starting solution for high res
    u_prolonged = prolong_displacement(...
        u_star, T_ML{i - 1}.s, T_ML{i}.s);
    
    % update objective function to high res
    f = @(u) objective_function(dist_fctn, reg_fctn, lambda, ...
        T_ML{i}.img, R_ML{i}.img, R_ML{i}.h, u);
    
    % more output
    str = sprintf('REGISTRATING AT LEVEL %d WITH RESOLUTION [%d, %d]', ...
        i, R_ML{i}.s(1), R_ML{i}.s(2));
    fprintf('\n%s%s%s\n', ...
        repmat('~', [1, floor(0.5 * (80 - length(str)))]), ...
        str, ...
        repmat('~', [1, ceil(0.5 * (80 - length(str)))]));

    u_star = newton_scheme(f, u_prolonged);
end

% evaluate output
u_star = reshape(u_star, [m * n, 2]);
T_u_star = evaluate_displacement(...
    T_ML{num_levels}.img, T_ML{num_levels}.h, u_star);

% compute grid g from displacement u
[cc_x, cc_y] = cell_centered_grid([m, n], h);
g = [cc_x(:), cc_y(:)] + u_star;
g = reshape(g, [m, n, 2]);

%% display results
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(1, 3, 1);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', R);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('reference R');

subplot(1, 3, 2);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
plot_grid(g, 4);
title('template T with displaced grid')

subplot(1, 3, 3);
image(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', T_u_star);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('transformed template T_u');