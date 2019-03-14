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

% demo script for mf_nn_registration_no_ref.m
clear all, close all, clc;

% % create data
% m = 100;    n = 100;    k = 8;
% data = dynamicTestImage(m, n, k);
% img = cell(k, 1);
% for i = 1 : k, img{i} = data(:, :, i); end
% % try rotated version
% for i = 1 : k, img{i} = imrotate(img{i}, 30, 'bilinear', 'crop'); end
% % set optimization parameters
% optPara.theta = 1;
% optPara.maxIter = 2000;
% optPara.tol = 1e-3;
% optPara.outerIter = 20;
% optPara.mu = 1e-1;
% optPara.nu_factor = 0.9;
% optPara.bc = 'linear';
% optPara.doPlots = true;

% load data
load('heart_mri.mat');
IDX = [3, 17, 32, 45, 60, 73, 90, 104];
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
optPara.mu = 1.5e-1;
optPara.nu_factor = 0.875;
optPara.bc = 'linear';
optPara.doPlots = true;

% call registration routine
tic;
[u, L] = mf_nn_registration_no_ref(img, optPara);
toc;

%% display results

% clean-up
close all;

% singular values
plot_sv(L);

% overview of all warped images
iter = 20;
img_u = display_results(img, u{iter}, [], L{iter});

% input, output and low rank components in comparison
figure;
colormap gray(256);
while true
    for i = 1 : k
        subplot(1, 3, 1);
        imshow(img{i}, [0 1], 'InitialMagnification', 'fit');
        title(sprintf('input T_%d', i));
        subplot(1, 3, 2);
        imshow(img_u{i}, [0 1], 'InitialMagnification', 'fit');
        title(sprintf('output T_%d(u_%d)', i, i));
        subplot(1, 3, 3);
        imshow(L{iter}(:, :, i), [0 1], 'InitialMagnification', 'fit');
        title(sprintf('output L_%d', i));
        waitforbuttonpress;
    end
end