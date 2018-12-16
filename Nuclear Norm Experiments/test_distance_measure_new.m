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
    numImg = 4;
    
    % downsample frames
    img = cell(numImg, 1);
    for i = 1 : numImg
        img{i} = conv2(A(:, :, i), ones(2) / 4);
        img{i} = img{i}(2 : 2 : end, 2 : 2 : end);
    end
    
else
    error('Missing file: ''respfilm1gray.mat''!');
end

% get image resolution etc.
[m, n] = size(img{1});
h = 1 ./ size(img{1});

% column major vectorization operator
vec = @(x) x(:);

%% PRECOMPUTE DISPLACED IMAGES AND CORRESPONDING DATA TERM VALUES

% directions of moving images
dir = [cos((0 : numImg - 2) * 2 * pi / (numImg - 1)); ...
    sin((0 : numImg - 2) * 2 * pi / (numImg - 1))]';

% placeholders for shifted images
numFrames = 101;
t = linspace(-1, 1, numFrames);
img_u = cell(numImg - 1, 1);
for i = 1 : numImg - 1
    img_u{i} = zeros(m, n, numFrames);
end

% get data term     mu * ||L||_* + ||L - I(u)||_1   by optimization over L
%   -> set parameters for Chambolle-Pock scheme
data_term = zeros(numFrames, 3);
theta = 1;
K = speye(m * n * numImg);              norm_K = 1;
tau = sqrt((1 - 1e-4) / norm_K ^ 2);    sigma = tau;
L0 = zeros(m * n * numImg, 1);          P0 = L0;
mu = 0.1;     G = @(L, c_flag) nuclear_norm(L, numImg, tau, mu, c_flag);

% column major image matrix (fixed first column = reference)
I = zeros(m * n, numImg);
I(:, 1) = vec(img{1});

for i = 1 : numFrames
    
    % evaluation of displacements
    for j = 1 : numImg - 1
        
        % warp image + store it
        u = t(i) * ones(m * n, 2) .* dir(j, :);
        img_u{j}(:, :, i) = evaluate_displacement(img{j}, h, u);
        
        % write warped image into corresponding I-column
        I(:, j + 1) = vec(img_u{j}(:, :, i));
        
    end
    
    % perform optimization
    F = @(L, c_flag) SAD(L, I(:), sigma, c_flag);
    [~, ~, primal_history, ~] = ...
        chambolle_pock(F, G, K, L0, P0, theta, tau, sigma);
    data_term(i, :) = primal_history(end, :);
    
end

%% PLOTTING

% get numImg different colors
c_map = hsv(numImg - 1);
C = cell(numImg - 1, 1);
for i = 1 : numImg - 1
    C{i} = ones(m, n, 3) .* reshape(c_map(i, :), [1 1 3]);
end

% get figure ready
figure('units', 'normalized', 'position', [0 0 1 1]);
colormap gray(256);
subplot(1, 2, 1);               axis image;
set(gca, 'YDir', 'reverse');    xlabel('---y-->');      ylabel('<--x---');
title(sprintf('I_1 vs. I_2(u_2), .., I_%d(u_%d)', numImg, numImg));

for i = 1 : numFrames
    
    subplot(1, 2, 1);
    imagesc(...
        'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
        'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
        'CData', img{1});
    hold on;
    for j = 1 : numImg - 1
        imagesc(...
            'YData', [h(1) * (1/2), h(1) * (m - (1/2))], ...
            'XData', [h(2) * (1/2), h(2) * (n - (1/2))], ...
            'CData', C{j}, ...
            'AlphaData', 0.75 * img_u{j}(:, :, i));
    end
    hold off;
    
    subplot(1, 2, 2);
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
        set(gca, 'Color', 0.6 * ones(3, 1));
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
    
    drawnow;        pause(0.1);
    
end