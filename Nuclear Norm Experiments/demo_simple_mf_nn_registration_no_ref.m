% demo script for simple_mf_nn_registration_no_ref.m

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
optPara.outerIter = 15;
optPara.mu = 2e-1;
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