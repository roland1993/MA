%% INITIALIZATION

% clean-up
clear all, clc, close all;

% make sure that interpolation routines are on search path
if ~exist('evaluate_displacement.m')
    addpath(genpath('..'));
end

% load respiratory data if available
if ~exist('respfilm1gray.mat')
    error('Missing file: ''respfilm1gray.mat''!');
end

% select experiment
fprintf('%sTESTING MEAN-FREE NUCLEAR NORM DISTANCE MEASURE%s\n', ...
    repmat('~', 16, 1), repmat('~', 17, 1));

answer = [];
while isempty(answer) || ...
        ~(answer == 1 || answer == 2 || answer == 3 || answer == 4)
    fprintf('\nCHOOSE EXPERIMENT FROM:\n\n');
    fprintf('\t1\tTRANSLATION\n');
    fprintf('\t2\tROTATION\n');
    fprintf('\t3\tZOOMING\n');
    fprintf('\t4\tSHEARING\n\n');
    answer = str2num(input('YOUR CHOICE: ','s'));
end

% load film and normalize to [0 1]
load('respfilm1gray.mat', 'A');
A = (A - min(A(:))) / (max(A(:)) - min(A(:)));

% number of frames to use
numImg = 2;
idx = floor(linspace(1, size(A, 3), numImg));

% downsample frames
img = cell(numImg, 1);
for i = 1 : numImg
    img{i} = conv2(A(:, :, idx(i)), ones(2) / 4);
    img{i} = img{i}(2 : 2 : end, 2 : 2 : end);
end

% get image resolution etc.
[m, n] = size(img{1});
h_img = 1 ./ size(img{1});

% column major vectorization operator
vec = @(x) x(:);

%% PRECOMPUTE WARPED IMAGES AND CORRESPONDING DATA TERM VALUES

% placeholders for warped images
numFrames = 31;
t = linspace(-1, 1, numFrames);
img_u = cell(numImg - 1, 1);
for i = 1 : numImg - 1
    img_u{i} = zeros(m, n, numFrames);
end

% set image region
omega = [0, 1, 0, 1];

% get image center
center = (omega([4, 2]) + omega([3, 1])) / 2;
[xx, yy] = cell_centered_grid(omega, [m, n]);
    
if answer == 1
    
    % direction of moving image
    dir = randn(1, 2);
    dir = dir / max(abs(dir));
    
elseif answer == 2
    
    % cut out rounded image disk    
    R = 0.375;
    c_dist = sqrt((xx(:) - center(1)) .^ 2 + (yy(:) - center(2)) .^ 2);
    tmp = (1 - c_dist / R);
    idx = tmp >= 0;
    tmp(idx) = tmp(idx) .^ 0.5;
    tmp(~idx) = 0;
    mask = reshape(tmp, [m, n]);
    for i = 1 : numImg
        img{i} = img{i} .* mask;
    end
    
end

% get grid step sizes
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% get data term by optimization over L
%   -> set parameters for Chambolle-Pock scheme
data_term = zeros(numFrames, 1);
theta = 1;
% K = [speye(m * n * numImg); speye(m * n * numImg)];
B = mean_free_operator(m, n, numImg);
K = [B; speye(m * n * numImg)];
norm_K = normest(K);
tau = sqrt(0.975 / norm_K ^ 2);
sigma = sqrt(0.975 / norm_K ^ 2);
L0 = zeros(m * n * numImg, 1);
P0 = zeros(2 * m * n * numImg, 1);
tol = 1e-3;
maxIter = 200;

% column major image matrix (fixed first column = reference)
I = zeros(m * n, numImg);
I(:, 1) = vec(img{1});

for i = 1 : numFrames
    
    % compute warped images
    for j = 1 : numImg - 1
                
        % warp image
        if answer == 1
            
            u = t(i) * ones(m * n, 2) .* dir(j, :);
            
        elseif answer == 2
            
            u = [xx(:), yy(:)] - center;
            alpha = t(i) * pi;
            u = u * [cos(alpha), sin(alpha); -sin(alpha), cos(alpha)];
            u = u + center;
            u = u - [xx(:), yy(:)];
            
        elseif answer == 3
            
            u = [xx(:), yy(:)] - center;
            u = (1 - t(i)) ^ 2 * u;
            u = u + center;
            u = u - [xx(:), yy(:)];
            
        elseif answer == 4
           
            u = [xx(:), yy(:)] - center;
            u(:, 2) = u(:, 2) + t(i) * u(:, 1);
            u = u + center;
            u = u - [xx(:), yy(:)];
            
        end
        
        % write warped image into corresponding I-column
        img_u{j}(:, :, i) = ...
            evaluate_displacement(img{j + 1}, h_img, u, omega);
        I(:, j + 1) = vec(img_u{j}(:, :, i));
        
    end
    
    % compute nu from nuclear norm of I
    [~, S, ~] = svd(I, 'econ');
    nu = 0.5 * sum(diag(S));
    G = @(x, c_flag) zero_function(x, c_flag);
    
    % optimization to find data term value
    F = @(y, c_flag) F_composite(y, numImg, sigma, nu, vec(I), c_flag);
    [Lstar, ~, primal_history, ~] = ...
        chambolle_pock(F, G, K, L0, P0, theta, tau, sigma, maxIter, tol);
    data_term(i) = primal_history(end, 1);

end

%% PLOTTING

% get different colors
c_map = hsv(numImg - 1);
C = cell(numImg - 1, 1);
for i = 1 : numImg - 1
    C{i} = ones(m, n, 3) .* reshape(c_map(i, :), [1 1 3]);
end

% get figure ready
figure;
colormap gray(256);
subplot(1, 2, 1);
set(gca, 'YDir', 'reverse');
xlabel('---y-->');
ylabel('<--x---');
title('I_1 vs. I_2(u)');
axis image;

for i = 1 : numFrames
    
    subplot(1, 2, 1);
    imagesc(...
        'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
        'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
        'CData', img{1});
    hold on;
    for j = 1 : numImg - 1
        imagesc(...
            'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
            'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
            'CData', C{j}, ...
            'AlphaData', img_u{j}(:, :, i));
    end
    hold off;
    
    subplot(1, 2, 2);
    if i == 1
        p1 = plot(1 : i, data_term(1 : i), 'y', 'LineWidth', 2);
        hold on;
        p4 = plot(i, data_term(i), 'ro', 'HandleVisibility', 'off');
        hold off;
        xlim([1, numFrames]);
        ylim([0, 1.25 * max(data_term(:))]);
        set(gca, 'Color', 0.6 * ones(3, 1));
            title('|| L - I(u) ||_1 + \delta_{||.||_* <= \nu} ( L )');
    else
        set(p1, 'XData', 1 : i, 'YData', data_term(1 : i));
        set(p4, 'XData', i, 'YData', data_term(i));
    end
    
    drawnow;
    pause(0.1);
    
end

%% LOCAL FUNCTION DEFINITIONS

function [res1, res2, res3] = ...
    F_composite(y, numImg, sigma, nu, b, conjugate_flag)

mn = numel(y) / (2 * numImg);
y1 = y(1 : numImg * mn);
y2 = y(numImg * mn + 1 : end);

[res1_F1, res2_F1, res3_F1] = ...
    nuclear_norm_constraint(y1, numImg, sigma, nu, conjugate_flag);

[res1_F2, res2_F2, res3_F2] = ...
    SAD(y2, b, sigma, conjugate_flag);

res1 = res1_F1 + res1_F2;
res2 = max([res2_F1, res2_F2]);
res3 = [res3_F1; res3_F2];

end