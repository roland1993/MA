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
    fprintf('\nNUCLEAR NORM AS:\n\n');
    fprintf('\t1\tSOFT CONSTRAINT\n');
    fprintf('\t2\tHARD CONSTRAINT\n');
    fprintf('\t3\tHARD CONSTRAINT MODIFIED\n\n');
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
    for i = 1 : numImg
        img{i} = img{i} .* mask;
    end
    
end

% get grid step sizes
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% get data term by optimization over L
%   -> set parameters for Chambolle-Pock scheme
data_term = zeros(numFrames, 3);
theta = 1;
K = speye(m * n * numImg);
norm_K = 1;
tau = sqrt((1 - 1e-4) / norm_K ^ 2);
sigma = tau;
L0 = zeros(m * n * numImg, 1);
P0 = L0;
tol = 5e-5;
maxIter = 200;

% track SSD for comparison
SSD = zeros(numFrames, 1);

% track TV as well
TV = zeros(numFrames, 1);

% define discrete gradient operator K for evaluation of TV
Dx = (1 / h_grid(1)) * spdiags([-ones(m, 1), ones(m, 1)], 0 : 1, m, m);
Dx(m, m) = 0;
Dy = (1 / h_grid(2)) * spdiags([-ones(n, 1), ones(n, 1)], 0 : 1, n, n);
Dy(n, n) = 0;
Gx = kron(speye(n), Dx);
Gy = kron(Dy, speye(m));
A = kron(speye(2), [Gx; Gy]);

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
    
    if version == 1
        
        % set parameter mu
        mu = 100;
        G = @(L, c_flag) nuclear_norm(L, numImg, tau, mu, c_flag);
        
    elseif version == 2
        
        % compute nu from nuclear norm of I
        [~, S, ~] = svd(I, 'econ');
        nu = 0.9 * sum(diag(S));
        G = @(L, c_flag) ...
            nuclear_norm_constraint(L, numImg, tau, nu, c_flag);
        
    else
        
        % compute nu from nuclear norm of I - R
        D = I - repmat(vec(img{1}), [1, numImg]);
        [~, S, ~] = svd(D, 'econ');
        nu = 0.9 * sum(diag(S));
        G = @(L, c_flag) ...
            nuclear_norm_constraint_mod(L, img{1}, tau, nu, c_flag);
        
    end
    
    % optimization to find data term value
    F = @(L, c_flag) SAD(L, I(:), sigma, c_flag);
    [~, ~, primal_history, ~] = ...
        chambolle_pock(F, G, K, L0, P0, theta, tau, sigma, maxIter, tol);
    data_term(i, :) = primal_history(end, 1 : 3);
    
    % compute SSD
    SSD(i) = sum((I(:, 1) - I(:, 2)) .^ 2);
    
    % compute TV
    TV(i) = TV_registration(A * u(:), []);
    
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
        p1 = plot(1 : i, data_term(1 : i, 1), 'y', 'LineWidth', 2);
        hold on;
        p4 = plot(i, data_term(i, 1), 'ro', 'HandleVisibility', 'off');
        if version == 1
            p2 = plot(1 : i, data_term(1 : i, 2), '--m');
            p3 = plot(1 : i, data_term(1 : i, 3), '--g');
            p5 = plot(i, data_term(i, 2), 'ro', 'HandleVisibility', 'off');
            p6 = plot(i, data_term(i, 3), 'ro', 'HandleVisibility', 'off');
        end
        hold off;
        xlim([1, numFrames]);
        ylim([0, 1.25 * max(data_term(:, 1))]);
        set(gca, 'Color', 0.6 * ones(3, 1));
        if version == 2
            title('|| L - I(u) ||_1 + \delta_{||.||_* <= \nu} ( L )');
        elseif version == 3
            title('|| L - I(u) ||_1 + \delta_{||.||_* <= \nu} ( L - R )');
        else
            title('\mu || L ||_* + || L - I(u) ||_1');
            legend('distance', '|| L - I(u) ||_1', '\mu || L ||_*', ...
                'Location', 'SouthOutside', 'Orientation', 'Horizontal');
        end
    else
        set(p1, 'XData', 1 : i, 'YData', data_term(1 : i, 1));
        set(p4, 'XData', i, 'YData', data_term(i, 1));
        if version == 1
            set(p2, 'XData', 1 : i, 'YData', data_term(1 : i, 2));
            set(p3, 'XData', 1 : i, 'YData', data_term(1 : i, 3));
            set(p5, 'XData', i, 'YData', data_term(i, 2));
            set(p6, 'XData', i, 'YData', data_term(i, 3));
        end
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
        title('TV || \nablau ||_1');
    else
        set(r1, 'XData', 1 : i, 'YData', TV(1 : i));
        set(r2, 'XData', i, 'YData', TV(i));
    end
    
    drawnow;
    pause(0.1);
    
end