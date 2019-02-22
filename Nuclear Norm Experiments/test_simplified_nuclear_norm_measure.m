%   test script for data terms
%       (1)     || [I_1(u_1), I_2] ||_*
%       (2)     || [I_1(u_1) - I_2] ||_*
%       (3)     || B * [I_1(u_1), I_2] ||_*

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
fprintf('%sTESTING NUCLEAR NORM DISTANCE MEASURE%s\n', ...
    repmat('~', 21, 1), repmat('~', 22, 1));

version = [];
while isempty(version) || ~(version == 1 || version == 2 || version == 3)
    fprintf('\nCHOOSE VERSION:\n\n');
    fprintf('\t1\tPURE NUCLEAR NORM\n');
    fprintf('\t2\tNUCLEAR NORM MINUS REFERENCE\n');
    fprintf('\t3\tMEAN-FREE NUCLEAR NORM\n\n');
    version = str2num(input('YOUR CHOICE: ','s'));
end

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
    for i = 1 : numImg, img{i} = img{i} .* mask; end
    
end

% get grid step sizes
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% track nuclear norm as data term
NN = zeros(numFrames, 1);

% track SSD for comparison
SSD = zeros(numFrames, 1);

% track TV as well
TV = zeros(numFrames, 1);

% define discrete gradient operator for evaluation of TV
D = finite_difference_operator(m, n, h_grid, 1, 'neumann');

% build column major image matrix
if version == 1
    I = zeros(m * n, numImg);
    I(:, 1) = vec(img{1});
elseif version == 2
    I = zeros(m * n, numImg - 1);
else
    I = zeros(m * n, numImg);
    I(:, 1) = vec(img{1});
end

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
        
        if version == 1 || version == 3
            I(:, j + 1) = vec(img_u{j}(:, :, i));
        else
            I(:, j) = vec(img_u{j}(:, :, i)) - vec(img{1});
        end
        
    end
    
    if version == 3
        mu = sum(I, 2) / size(I, 2);
        [~, SV, ~] = svd(I - mu, 'econ');
    else
        [~, SV, ~] = svd(I, 'econ');
    end
    
    % calculate nuclear norm
    NN(i) = sum(diag(SV));
    
    % compute SSD
    SSD(i) = sum((vec(img{1}) - vec(img_u{1}(:, :, i))) .^ 2);
    
    % compute TV
    TV(i) = TV_registration(D * u(:), []);
    
end

%% PLOTTING

% get different colors
c_map = hsv(numImg - 1);
C = cell(numImg - 1, 1);
for i = 1 : numImg - 1
    C{i} = ones(m, n, 3) .* reshape(c_map(i, :), [1 1 3]);
end

% get figure ready
figure('units', 'normalized', 'position', [0 0 1 1]);
colormap gray(256);
subplot(2, 2, 1);
set(gca, 'YDir', 'reverse');
xlabel('---y-->');
ylabel('<--x---');
title('I_1 vs. I_2(u)');
axis image;

for i = 1 : numFrames
    
    subplot(2, 2, 1);
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
    
    subplot(2, 2, 2);
    if i == 1
        p1 = plot(1 : i, NN(1 : i), 'y', 'LineWidth', 2);
        hold on;
        p4 = plot(i, NN(i), 'ro', 'HandleVisibility', 'off');
        hold off;
        xlim([1, numFrames]);
        ylim([0, 1.25 * max(NN)]);
        set(gca, 'Color', 0.6 * ones(3, 1));
        if version == 1
            title('|| [I_1, I_2(u)] ||_* ');
        elseif version == 2
            title('|| [I_1 - I_2(u)] ||_* ');
        else
            title('|| [I_1 - \mu, I_2(u) - \mu] ||_* ');
        end
            
    else
        set(p1, 'XData', 1 : i, 'YData', NN(1 : i));
        set(p4, 'XData', i, 'YData', NN(i));
    end
    
    subplot(2, 2, 3);
    if i == 1
        q1 = plot(1 : i, SSD(1 : i), 'y', 'LineWidth', 2);
        hold on;
        q2 = plot(i, SSD(i), 'ro');
        hold off;
        xlim([1, numFrames]);
        ylim([0, 1.25 * max(SSD)]);
        set(gca, 'Color', 0.6 * ones(3, 1));
        title('SSD || I_1 - I_2(u) ||_2^{ 2}');
    else
        set(q1, 'XData', 1 : i, 'YData', SSD(1 : i));
        set(q2, 'XData', i, 'YData', SSD(i));
    end
    
    subplot(2, 2, 4);
    if i == 1
        r1 = plot(1 : i, TV(1 : i), 'y', 'LineWidth', 2);
        hold on;
        r2 = plot(i, TV(i), 'ro');
        hold off;
        xlim([1, numFrames]);
        if max(TV) > 0
            ylim([0, 1.25 * max(TV)]);
        end
        set(gca, 'Color', 0.6 * ones(3, 1));
        title('Total Variation || \nablau ||_{2,1}');
    else
        set(r1, 'XData', 1 : i, 'YData', TV(1 : i));
        set(r2, 'XData', i, 'YData', TV(i));
    end
    
    drawnow;
    pause(0.1);
    
end