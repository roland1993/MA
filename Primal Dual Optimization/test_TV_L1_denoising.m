%% initialization

% clean-up
clear all, close all, clc;

% read sample image
img = double(imread('westconcordorthophoto.png'));

% normalize image to [0, 1]
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% create noisy version of input image
p = 0.25;
img_noisy = imnoise(img, 'salt & pepper', p);

% get image size & set grid width / height
[m, n] = size(img);
h = [1, 1];

% display input data
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap gray(256);

subplot(2, 2, 1);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', img);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('input image img');

subplot(2, 2, 2);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', img_noisy);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title(['img\_noisy with ', num2str(p * 100),'% salt & pepper noise']);

%% setup and optimization

% define discrete gradient operator K
Dx = (1 / h(1)) * spdiags([-ones(m, 1), ones(m, 1)], 0 : 1, m, m);
Dx(m, m) = 0;
Dy = (1 / h(2)) * spdiags([-ones(n, 1), ones(n, 1)], 0 : 1, n, n);
Dy(n, n) = 0;
Gx = kron(speye(n), Dx);    Gy = kron(Dy, speye(m));
K = [Gx; Gy];

% upper bound on spectral norm of K
L_squared = 4 * (1 / h(1) ^ 2 + 1 / h(2) ^ 2);

% set parameters of optimization scheme 
u0 = zeros(m * n, 1);
v0 = zeros(m * n * 2, 1);
theta = 1;
tau = 0.02;
sigma = 1 / (L_squared * tau);
maxIter = 250;

% function handles for data term and regularizer
lambda = 1.75;
G = @(u) SAD(u, img_noisy(:), lambda, tau);
F = @(v) TV(v, sigma);

% perform optimization
[u_star, v_star] = ...
    chambolle_pock(F, G, K, u0, v0, theta, tau, sigma, maxIter);
img_denoise = reshape(u_star, size(img));

%% display results

subplot(2, 2, 3);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', img_denoise);
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('denoising result img\_denoise');

subplot(2, 2, 4);
imagesc(...
    'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
    'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
    'CData', abs(img_denoise - img));
axis image;     set(gca, 'YDir', 'reverse');
colorbar;
xlabel('---y-->');      ylabel('<--x---');
title('absolute difference |img - img\_denoise|');