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

exp_begin();

% demo script for mf_nn_registration_no_ref_ml.m
clear all, close all, clc;

% normalization to [0, 1]
normalize = @(y) (y - min(y(:))) / (max(y(:)) - min(y(:)));

% select data set from {'BlinkingArrow', 'CarTruck', 'CrossingCars', ...
%   'FlyingSnow', 'NightAndSnow', 'RainBlur', 'RainFlares', ...
%   'ReflectingCar', 'ShadowOnTruck', 'SunFlare', 'WetAutobahn'}
data_set = 'FlyingSnow';
data_path = sprintf('../Data/ChallengingSequences/%s/sequence', data_set);

switch data_set
    
    case 'FlyingSnow'
        
        % load + downsample images
        IDX = 1 : 10;
        k = numel(IDX);
        img = cell(1, k);
%         factor = 2;
        for i = 1 : k
            img{i} = normalize(double( ...
                sub_imread(sprintf('%s/%06d_0.pgm', data_path, IDX(i)))));
%             img{i} = conv2(img{i}, ones(factor) / factor ^ 2, 'same');
%             img{i} = img{i}(1 : factor : end, 1 : factor : end);
        end
        
        % optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [20 1];
        optPara.mu = 5e-2;
        optPara.nu_factor = [0.95 1];
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
        % select reference
        ref_idx = 4 - IDX(1) + 1;
        
    case 'RainBlur'
        
        % load + downsample images
        IDX = 10 : 15;
        k = numel(IDX);
        img = cell(1, k);
        factor = 2;
        for i = 1 : k
            img{i} = normalize(double( ...
                sub_imread(sprintf('%s/%06d_0.pgm', data_path, IDX(i)))));
            img{i} = conv2(img{i}, ones(factor) / factor ^ 2, 'same');
            img{i} = img{i}(1 : factor : end, 1 : factor : end);
        end
        
        % optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [16 2];
        optPara.mu = 5e-2;
        optPara.nu_factor = [0.95 0.95];
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
        % select reference
        ref_idx = 11 - IDX(1) + 1;
        
    case 'BlinkingArrow'
        
        % load + downsample images
        IDX = 1 : 10;
        k = numel(IDX);
        img = cell(1, k);
        factor = 2;
        for i = 1 : k
            img{i} = normalize(double( ...
                sub_imread(sprintf('%s/%06d_0.pgm', data_path, IDX(i)))));
            img{i} = conv2(img{i}, ones(factor) / factor ^ 2, 'same');
            img{i} = img{i}(1 : factor : end, 1 : factor : end);
        end
        
        % optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [16 2];
        optPara.mu = 4.5e-2;
        optPara.nu_factor = [0.95 0.95];
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
        % select reference
        ref_idx = 3 - IDX(1) + 1;
        
    otherwise
        
        % todo
        
end

% call registration routine
tic;
[u, L, SV_history] = mf_nn_registration_fix_ref_ml(img, ref_idx, optPara);
toc;

%% display results

img_u = cell(k, 1);
for i = 1 : k
    img_u{i} = evaluate_displacement(img{i}, [1, 1], ...
        u{end, optPara.outerIter(2)}(:, :, i));
end

exp_save('data');
exp_end();

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