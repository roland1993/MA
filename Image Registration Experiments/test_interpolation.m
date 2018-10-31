%% initialization
clear all, close all, clc;

% generate random image data
m = 3;
n = 4;
h = [1, 1];
X = randi(256, m, n) - 1;

%% display image data
figure;
colormap gray(256);
hold on;
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(X));
axis xy;
colorbar;
xlabel('---x-->');
ylabel('---y-->');

% plot cell centered grid
[x, y] = cell_centered_grid([m, n], h);
p = [x(:), y(:)];
plot(p(:, 1), p(:, 2), 'r+');
title('image X with cell centered grid');

%% interpolate X over cell centered grid (... should yield X)
X_interpol = bilinear_interpolation(X, h, p);
X_interpol = reshape(X_interpol, [m, n]);
fprintf('||X - X_interpol|| = %.3e\n', norm(X(:) - X_interpol(:)));

%% randomize small displacement to test evaluate_displacement
u = 0.1 * randn(m * n, 2);
p_displaced = p + u;
X_u = evaluate_displacement(X, h, u);

% display results
figure;
colormap gray(256);

subplot(1, 2, 1);
hold on;
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(X));
axis xy;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
plot(p_displaced(:, 1), p_displaced(:, 2), 'r+');
title('image X with displaced grid');

subplot(1, 2, 2);
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(X_u));
axis xy;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
title('image X interpolated at displaced grid');