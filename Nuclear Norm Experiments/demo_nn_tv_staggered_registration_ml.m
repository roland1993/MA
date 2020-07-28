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

% choose dataset from {synthetic, heart}
dataset = 'synthetic';

switch dataset
    case 'synthetic'
        
        % generate data
        m = 200;    n = 200;    k = 10;
        [data, LM_data] = dynamicTestImage(m, n, k);
        img = cell(k, 1);
        LM = cell(k, 1);
        for i = 1 : k
            img{i} = data(:, :, i);
            LM{i} = LM_data(:, :, i);
        end
        omega = [0, m, 0, n];
        
        % set optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [16 2];
        optPara.mu = 2e-1;
        optPara.nu_factor = [0.9 0.9];
        optPara.doPlots = true;
        
    case 'heart'
        
        % load data
        load('heart_mri.mat');
        k = length(IDX);
        
        % downsampling
        img = cell(k, 1);
        factor = 2;
        for i = 1 : k
            img{i} = conv2(data(:, :, IDX(i)), ...
                ones(factor) / factor ^ 2, 'same');
            img{i} = img{i}(1 : factor : end, 1 : factor : end);
        end
        [m, n] = size(img{1});
        omega = [0, m, 0, n];
        
        % adjust scaling of landmarks
        LM = cell(1, k);
        for i = 1 : k
            LM{i} = [m n] .* LM_IDX{i};
        end
        
        % set optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [16 2];
        optPara.mu = 1.25e-1;
        optPara.nu_factor = [0.95 0.95];
        optPara.bc = 'neumann';
        optPara.doPlots = false;
        
    otherwise
        error('No such dataset!');
        
end

% call registration routine
tic;
[uStar, LStar, SV_history] = nn_tv_staggered_registration_ml(img, optPara);
toc;

% evaluate results
img_u = cell(k, 1);
LM_transformed = cell(k, 1);

for i = 1 : k
    
    % get transformed images
    img_u{i} = evaluate_displacement(img{i}, [1, 1], uStar(:, :, i));
    
    % transform landmarks
    LM_transformed{i} = ...
        landmark_transform(LM{i}, reshape(uStar(:, :, i), [m n 2]), omega);
    
end

% landmark accuracy in terms of mean distance to mean lm position
LM_acc = landmark_accuracy(LM);
LM_transformed_acc = landmark_accuracy(LM_transformed);

% input, output and low rank components in comparison
figure;
colormap gray(256);

while true
    for i = 1 : k
        
        subplot(1, 3, 1);
        imshow(img{i}, [], 'InitialMagnification', 'fit');
        hold on
        scatter(LM{i}(:, 2), LM{i}(:, 1), 'bo', 'MarkerFaceColor', 'red');
        hold off
        title(sprintf('input T_{%d}', i));
        
        subplot(1, 3, 2);
        imshow(img_u{i}, [], 'InitialMagnification', 'fit');
        hold on
        scatter(LM_transformed{i}(:, 2), LM_transformed{i}(:, 1), ...
            'bo', 'MarkerFaceColor', 'red');
        hold off
        title(sprintf('output T_{%d}(u_{%d})', i, i));
        
        subplot(1, 3, 3);
        imshow(LStar(:, :, i), [], 'InitialMagnification', 'fit');
        title(sprintf('output L_{%d}', i));
        
        waitforbuttonpress;
        
    end
end