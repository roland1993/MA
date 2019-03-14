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

% demo script for simple_mf_nn_registration.m
clear all, close all, clc;

% create data
m = 100;    n = 100;    k = 8;
data = dynamicTestImage(m, n, k + 1);
img = cell(k + 1, 1);
for i = 1 : (k + 1), img{i} = data(:, :, i); end

% find reference
[~, refIdx] = ...
    min(sum(reshape((data - mean(data, 3)) .^ 2, m * n, k + 1)));
IDX = 1 : k + 1;
IDX(refIdx) = 0;
IDX(refIdx + 1 : end) = IDX(refIdx + 1 : end) - 1;

% set optimization parameters
optPara.theta = 1;
optPara.maxIter = 2000;
optPara.tol = 1e-3;
optPara.outerIter = 15;
optPara.mu = 2e-1;
optPara.nu_factor = 0.9;
optPara.bc = 'linear';
optPara.doPlots = true;

% call registration routine
tic;
u = simple_mf_nn_registration(img, refIdx, optPara);
toc;

% display results
display_results(img, u{end}, refIdx);

% evaluate displacments and plot singular values
I = cell(optPara.outerIter, 1);
for i = 1 : optPara.outerIter
    I{i} = zeros(m, n, k + 1);
    for j = 1 : (k + 1)
        if IDX(j) == 0
            I{i}(:, :, j) = img{refIdx};
        else
            I{i}(:, :, j) = evaluate_displacement( ...
                img{j}, [1 1], u{i}(:, :, IDX(j)));
        end
    end
end
plot_sv(I);

%
figure;
colormap gray(256);
while true
    for i = 1 : (k + 1)
        subplot(1, 2, 1);
        imshow(img{i}, [0 1], 'InitialMagnification', 'fit');
        subplot(1, 2, 2);
        imshow(I{end}(:, :, i), [0 1], 'InitialMagnification', 'fit');
        waitforbuttonpress;
    end
end