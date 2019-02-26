% demo script for mf_nn_registration_no_ref.m

clear all, close all, clc;

% create data
m = 100;    n = 100;    k = 8;
data = dynamicTestImage(m, n, k);
img = cell(k, 1);
for i = 1 : k, img{i} = data(:, :, i); end

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
[u, L] = mf_nn_registration_no_ref(img, optPara);
toc;

% display results
img_u = display_results(img, u{end}, [], L{end});
plot_sv(L);

% 
figure;
colormap gray(256);
while true
    for i = 1 : k
        subplot(1, 3, 1);
        imshow(img{i}, [0 1], 'InitialMagnification', 'fit');
        subplot(1, 3, 2);
        imshow(img_u{i}, [0 1], 'InitialMagnification', 'fit');
        subplot(1, 3, 3);
        imshow(L{end}(:, :, i), [0 1], 'InitialMagnification', 'fit');
        waitforbuttonpress;
    end
end