%--------------------------------------------------------------------------
% This file is part of my master's thesis entitled
%           'Low rank- and sparsity-based image registration'
% For the whole project see
%           https://github.com/roland1993/MA
% If you have questions contact me at
%           roland.haase [at] student.uni-luebeck [dot] de
% Source code is provided under the
%           MIT Open Source License
%--------------------------------------------------------------------------

% demo script for mf_nn_registration_no_ref_ml.m
clear all, close all, clc;

%
normalize = @(y) (y - min(y(:))) / (max(y(:)) - min(y(:)));

% choose dataset from {synthetic, heart, kidney}
dataset = 'kidney';

switch dataset
    case 'synthetic'
        
        % generate data
        m = 200;    n = 200;    k = 8;
        data = dynamicTestImage(m, n, k);
        img = cell(k, 1);
        for i = 1 : k, img{i} = data(:, :, i); end
        
        % set optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [15 2];
        optPara.mu = 2e-1;
        optPara.nu_factor = [0.9 0.9];
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
    case 'heart'
        
        % load data
        load('heart_mri.mat');
        IDX = [ 003, 010, 017, ...
                032, 039, 045, ...
                060, 066, 073, ...
                090, 096, 104, ...
                121, 126, 134, ...
                154, 159, 167, ...
                194, 197, 206];
        k = length(IDX);
        
        % downsampling
        img = cell(k, 1);
        factor = 2;
        for i = 1 : k
            img{i} = conv2(data(:, :, IDX(i)), ...
                ones(factor) / factor ^ 2, 'same');
            img{i} = img{i}(1 : factor : end, 1 : factor : end);
        end
        
        % set optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [16 2];
        optPara.mu = 1.25e-1;
        optPara.nu_factor = [0.95 0.95];
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
    case 'kidney'
        
        % load data
        load('respfilm1gray.mat');
        k = 12;
        IDX = round(linspace(1, size(A, 3)/ 3, k));
        img = cell(1, k);
        for i = 1 : k
            img{i} = normalize(A(:, :, IDX(i)));
        end
        
        % downsampling
        factor = 2;
        for i = 1 : k
            img{i} = conv2(img{i}, ones(factor) / factor ^ 2, 'same');
            img{i} = img{i}(1 : factor : end, 1 : factor : end);
        end
        
        % set optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [15 2];
        optPara.mu = 1.25e-1;
        optPara.nu_factor = [0.9 1];
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
    otherwise
        error('No such dataset!');
        
end

% call registration routine
tic;
[u, L, SV_history] = mf_nn_registration_no_ref_ml(img, optPara);
toc;

%% display results

img_u = cell(k, 1);
for i = 1 : k
    img_u{i} = evaluate_displacement(img{i}, [1, 1], u{end, 1}(:, :, i));
end

% input, output and low rank components in comparison
figure;
colormap gray(256);
while true
    for i = 1 : k
        subplot(1, 3, 1);
        imshow(img{i}, [], 'InitialMagnification', 'fit');
        title(sprintf('input T_{%d}', i));
        subplot(1, 3, 2);
        imshow(img_u{i}, [], 'InitialMagnification', 'fit');
        title(sprintf('output T_{%d}(u_{%d})', i, i));
        subplot(1, 3, 3);
        imshow(L{end, 1}(:, :, i), [], 'InitialMagnification', 'fit');
        title(sprintf('output L_{%d}', i));
        waitforbuttonpress;
    end
end