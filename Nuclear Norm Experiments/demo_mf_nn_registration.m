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

% demo script for mf_nn_registration.m
clear all, close all, clc;

% create data
m = 100;    n = 100;    k = 8;
data = dynamicTestImage(m, n, k + 1);
img = cell(k + 1, 1);
for i = 1 : (k + 1), img{i} = data(:, :, i); end

% find reference
[~, refIdx] = ...
    min(sum(reshape((data - mean(data, 3)) .^ 2, m * n, k + 1)));

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
[u, L] = mf_nn_registration(img, refIdx, optPara);
toc;

% display results
img_u = display_results(img, u{end}, refIdx, L{end});
plot_sv(L);

% 
figure;
colormap gray(256);
while true
    for i = 1 : (k + 1)
        subplot(1, 3, 1);
        imshow(img{i}, [0 1], 'InitialMagnification', 'fit');
        subplot(1, 3, 2);
        imshow(img_u{i}, [0 1], 'InitialMagnification', 'fit');
        subplot(1, 3, 3);
        imshow(L{end}(:, :, i), [0 1], 'InitialMagnification', 'fit');
        waitforbuttonpress;
    end
end