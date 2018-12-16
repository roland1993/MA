%% CLEAN-UP + INITIALIZATION

clear all, close all, clc;

% make sure that interpolation routines are on search path
if ~exist('evaluate_displacement.m')
    addpath(genpath('..'));
end

% load respiratory data
if exist('respfilm1gray.mat')
    
    % load film and normalize to [0 1]
    load('respfilm1gray.mat', 'A');
    A = (A - min(A(:))) / (max(A(:)) - min(A(:)));
    
    % number of frames to use
    numImg = 3;
    
    % downsample frames by some degree
    img1 = conv2(A(:, :, 1), ones(2) / 4);
    img1 = img1(2 : 2 : end, 2 : 2 : end);
    img2 = conv2(A(:, :, 2), ones(2) / 4);
    img2 = img2(2 : 2 : end, 2 : 2 : end);
    img3 = conv2(A(:, :, 3), ones(2) / 4);
    img3 = img3(2 : 2 : end, 2 : 2 : end);
    
else
    error('Missing file: ''respfilm1gray.mat''!');
end

% get image resolution etc.
[m, n] = size(img1);
h = 1 ./ size(img1);

% column major vectorization operator
vec = @(x) x(:);

%% PRECOMPUTE DISPLACED IMAGES AND CORRESPONDING DATA TERM VALUES

% directions of moving images (orthogonal)
dir1 = randn(1, 2);             dir1 = dir1 / max(abs(dir1));
dir2 = [dir1(2), -dir1(1)];

% placeholders for shifted images
numFrames = 101;
t = linspace(-1, 1, numFrames);
img2_u = zeros(m, n, numFrames);
img3_u = zeros(m, n, numFrames);

% get data term     mu * ||L||_* + ||L - I(u)||_1   by optimization over L
%   -> set parameters for Chambolle-Pock scheme
data_term = zeros(numFrames, 3);
theta = 1;
K = speye(m * n * numImg);              norm_K = 1;
tau = sqrt((1 - 1e-4) / norm_K ^ 2);    sigma = tau;
L0 = zeros(m * n * numImg, 1);          P0 = L0;
mu = 0.1;     G = @(L, c_flag) nuclear_norm(L, numImg, tau, mu, c_flag);

% placeholders for minimizers
L1 = zeros(m, n, numFrames);
L2 = zeros(m, n, numFrames);
L3 = zeros(m, n, numFrames);

for i = 1 : numFrames
    
    % evaluation of displacements
    u1 = t(i) * ones(m * n, 2) .* dir1;
    img2_u(:, :, i) = evaluate_displacement(img2, h, u1);
    u2 = t(i) * ones(m * n, 2) .* dir2;
    img3_u(:, :, i) = evaluate_displacement(img3, h, u2);
    
    % optimization to find data term value
    I = [vec(img1), vec(img2_u(:, :, i)), vec(img3_u(:, :, i))];
    F = @(L, c_flag) SAD(L, I(:), sigma, c_flag);
    [L, ~, primal_history] = ...
        chambolle_pock(F, G, K, L0, P0, theta, tau, sigma);
    data_term(i, :) = primal_history(end, :);
    L = reshape(L, [], numImg);
    
    % store minimizer image-wise
    L1(:, :, i) = reshape(L(:, 1), [m n]);
    L2(:, :, i) = reshape(L(:, 2), [m n]);
    L3(:, :, i) = reshape(L(:, 3), [m n]);
    
end

%% PLOTTING

% display moving images as transparent overlays (in red and green)
red = uint8(cat(3, 255 * ones(m, n), zeros(m, n), zeros(m, n)));
green = uint8(cat(3, zeros(m, n), 255 * ones(m, n), zeros(m, n)));

figure('units', 'normalized', 'position', [0 0 1 1]);
colormap gray(256);

subplot(2, 3, 1);               axis image;
set(gca, 'YDir', 'reverse');    xlabel('---y-->');      ylabel('<--x---');
title('I_1  |  I_2  |  I_3');

subplot(2, 3, 4);               axis image;
set(gca, 'YDir', 'reverse');    xlabel('---y-->');      ylabel('<--x---');
title('L_1');

subplot(2, 3, 5);               axis image;
set(gca, 'YDir', 'reverse');    xlabel('---y-->');      ylabel('<--x---');
title('L_2');

subplot(2, 3, 6);               axis image;
set(gca, 'YDir', 'reverse');    xlabel('---y-->');      ylabel('<--x---');
title('L_3');

for i = 1 : numFrames
    
    subplot(2, 3, 1);
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', img1);
    hold on;
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', red, ...
        'AlphaData', 0.75 * img2_u(:, :, i));
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', green, ...
        'AlphaData', 0.75 * img3_u(:, :, i));
    hold off;
    
    subplot(2, 3, 2);
    if i == 1
        p1 = plot(1 : i, data_term(1 : i, 1), 'y', 'LineWidth', 2);
        hold on;
        p2 = plot(1 : i, data_term(1 : i, 2), '--m');
        p3 = plot(1 : i, data_term(1 : i, 3), '--g');
        p4 = plot(i, data_term(i, 1), 'ro');
        p5 = plot(i, data_term(i, 2), 'ro');
        p6 = plot(i, data_term(i, 3), 'ro');
        hold off;
        xlim([1, numFrames]);   ylim([0, 1.25 * max(data_term(:, 1))]);
        set(gca, 'Color', [0.8 0.8 0.8]);
        title('\mu || L ||_* + || L - I(u) ||_1');
        legend('distance', '|| L - I(u) ||_1', '\mu || L ||_*', ...
            'Location', 'SouthOutside', 'Orientation', 'Horizontal');
    else
        set(p1, 'XData', 1 : i, 'YData', data_term(1 : i, 1));
        set(p2, 'XData', 1 : i, 'YData', data_term(1 : i, 2));
        set(p3, 'XData', 1 : i, 'YData', data_term(1 : i, 3));
        set(p4, 'XData', i, 'YData', data_term(i, 1));
        set(p5, 'XData', i, 'YData', data_term(i, 2));
        set(p6, 'XData', i, 'YData', data_term(i, 3));
    end
    
    subplot(2, 3, 4);
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', L1(:, :, i));
    
    subplot(2, 3, 5);
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', L2(:, :, i));
    
    subplot(2, 3, 6);
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', L3(:, :, i));
    
    drawnow;        pause(0.1);
    
end