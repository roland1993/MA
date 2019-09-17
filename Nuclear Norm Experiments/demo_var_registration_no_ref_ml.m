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
        optPara.mu = 5e-2;
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
    case 'heart'
        
        % load data
        load('heart_mri.mat');
        IDX = [3, 10, 17, 32, 39, 45, 60, 66, 73, 90, 96, 104];
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
        optPara.outerIter = [15 2];
        optPara.mu = 2.5e-2;
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
    case 'kidney'
        
        % load data
        load('respfilm1gray.mat');
        k = 8;
        IDX = round(linspace(1, size(A, 3)/ 3, k));
        img = cell(1, k);
        for i = 1 : k
            img{i} = A(:, :, IDX(i));
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
        optPara.mu = 2e-2;
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
    otherwise
        error('No such dataset!');
        
end

% call registration routine
tic;
u = var_registration_no_ref_ml(img, optPara);
toc;

%% display results

img_u = cell(k, 1);
for i = 1 : k
    img_u{i} = evaluate_displacement(img{i}, [1, 1], ...
        u{end, optPara.outerIter(2)}(:, :, i));
end

% input, output and low rank components in comparison
figure;
colormap gray(256);
while true
    for i = 1 : k
        subplot(1, 2, 1);
        imshow(img{i}, [], 'InitialMagnification', 'fit');
        title(sprintf('input T_{%d}', i));
        subplot(1, 2, 2);
        imshow(img_u{i}, [], 'InitialMagnification', 'fit');
        title(sprintf('output T_{%d}(u_{%d})', i, i));
        waitforbuttonpress;
    end
end