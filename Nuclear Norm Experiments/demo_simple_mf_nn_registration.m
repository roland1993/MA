clear all, close all, clc;

% create data
m = 100;    n = 100;    k = 6;
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
optPara.outerIter = 15;
optPara.mu = 1e-1;
optPara.nu_factor = 0.9;
optPara.bc = 'linear';
optPara.doPlots = true;

% call registration routine
tic;
u = simple_mf_nn_registration(img, refIdx, optPara);
toc;

% display results
img_u = display_results(img, refIdx, u{end});

% TODO: plot singular values
I = zeros(m, n, k + 1);
for i = 1 : (k + 1), I(:, :, i) = img_u{i}; end
plot_sv(I);