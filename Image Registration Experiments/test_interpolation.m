%% initialization
clear all, close all, clc;

% generate random image data
m = 3;
n = 4;
h = [1, 1];
img = randi(256, m, n) - 1;

%% display image data
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);
hold on;
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(img));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');

% plot cell centered grid
[x, y] = cell_centered_grid([m, n], h);
p = [x(:), y(:)];
plot_grid(reshape(p, [m, n, 2]));
title('image img with cell centered grid');

%% interpolate img over cell centered grid (... should yield img)
img_interpol = bilinear_interpolation(img, h, p);
img_interpol = reshape(img_interpol, [m, n]);
fprintf('||img - img_interpol|| = %.3e\n', norm(img(:) - img_interpol(:)));

%% randomize small displacement to test evaluate_displacement
u = 0.1 * randn(m * n, 2);
p_displaced = p + u;
img_u = evaluate_displacement(img, h, u);

% display results
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(1, 2, 1);
hold on;
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(img));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
plot_grid(reshape(p_displaced, [m, n, 2]));
title('image img with displaced grid');

subplot(1, 2, 2);
image(...
    'Xdata', [h(1) / 2, (n - (1 / 2)) * h(1)], ...
    'YData', [h(2) / 2, (m - (1 / 2)) * h(2)], ...
    'CData', flipud(img_u));
axis xy;
axis image;
colorbar;
xlabel('---x-->');
ylabel('---y-->');
title('image img interpolated at displaced grid');