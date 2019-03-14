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

% demo script for simple_mf_nn_registration_no_ref.m
clear all, close all, clc;

% % create data
% m = 100;    n = 100;    k = 8;
% data = dynamicTestImage(m, n, k);
% img = cell(k, 1);
% for i = 1 : k, img{i} = data(:, :, i); end
% 
% % set optimization parameters
% optPara.theta = 1;
% optPara.maxIter = 2000;
% optPara.tol = 1e-3;
% optPara.outerIter = 15;
% optPara.mu = 2e-1;
% optPara.nu_factor = 0.9;
% optPara.bc = 'linear';
% optPara.doPlots = true;

% load data
load('heart_mri.mat');
IDX = [3, 32, 60, 90, 17, 45, 73, 104];
k = length(IDX);

% downsampling
img = cell(k, 1);
factor = 4;
for i = 1 : k
    tmp = conv2(data(:, :, IDX(i)), ones(factor) / factor ^ 2, 'same');
    img{i} = tmp(1 : factor : end, 1 : factor : end);
end

% set optimization parameters
optPara.theta = 1;
optPara.maxIter = 2000;
optPara.tol = 1e-3;
optPara.outerIter = 20;
optPara.mu = 1e-1;
optPara.nu_factor = 0.9;
optPara.bc = 'linear';
optPara.doPlots = true;

% call registration routine
tic;
u = simple_mf_nn_registration_no_ref(img, optPara);
toc;

% display results
display_results(img, u{end});

% evaluate displacments and plot singular values
m = size(img{1}, 1);   n = size(img{1}, 2);
I = cell(optPara.outerIter, 1);
for i = 1 : optPara.outerIter
    I{i} = zeros(m, n, k);
    for j = 1 : k
        I{i}(:, :, j) = evaluate_displacement( ...
            img{j}, [1 1], u{i}(:, :, j));
    end
end
plot_sv(I);

%
figure;
colormap gray(256);
while true
    for i = 1 : k
        subplot(1, 2, 1);
        imshow(img{i}, [0 1], 'InitialMagnification', 'fit');
        subplot(1, 2, 2);
        imshow(I{end}(:, :, i), [0 1], 'InitialMagnification', 'fit');
        waitforbuttonpress;
    end
end