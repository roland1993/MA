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

exp_begin();

% choose dataset from {synthetic, heart, kidney}
dataset = 'heart';

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
        optPara.mu = 1e-1;
        optPara.bc = 'neumann';
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
        for i = 1 : k
            LM{i} = [m n] .* LM_IDX2{i};
        end
        
        % set optimization parameters
        optPara.theta = 1;
        optPara.maxIter = 2000;
        optPara.tol = 1e-3;
        optPara.outerIter = [16 2];
        optPara.mu = 6.5e-2;
        optPara.bc = 'neumann';
        optPara.doPlots = true;
        
%     case 'kidney'
%         
%         % load data
%         load('respfilm1gray.mat');
%         k = 8;
%         IDX = round(linspace(1, size(A, 3)/ 3, k));
%         img = cell(1, k);
%         for i = 1 : k
%             img{i} = A(:, :, IDX(i));
%         end
%         
%         % downsampling
%         factor = 2;
%         for i = 1 : k
%             img{i} = conv2(img{i}, ones(factor) / factor ^ 2, 'same');
%             img{i} = img{i}(1 : factor : end, 1 : factor : end);
%         end
%         
%         % set optimization parameters
%         optPara.theta = 1;
%         optPara.maxIter = 2000;
%         optPara.tol = 1e-3;
%         optPara.outerIter = [15 2];
%         optPara.mu = 4e-2;
%         optPara.bc = 'neumann';
%         optPara.doPlots = true;
        
    otherwise
        error('No such dataset!');
        
end

% save name of method for later
optPara.method = mfilename;

% call registration routine
tic;
u = var_registration_no_ref_ml(img, optPara);
toc;

exp_end();
exp_save('data');

% fetch results
uStar = u{end, optPara.outerIter(2)};

% evaluate results
[xx, yy] = cell_centered_grid(omega, [m, n]);
p = [xx(:), yy(:)];
img_u = cell(k, 1);
LM_transformed = cell(k, 1);

for i = 1 : k
    
    % get transformed images
    img_u{i} = evaluate_displacement(img{i}, [1, 1], uStar(:, :, i));

    % get transformed grid
    g = p + uStar(:, :, i);
    
    for j = 1 : size(LM{i}, 1)
        
        % find closest point in transformed grid to current landmark
        %   -> approximate inversion of (id + u)
        [~, min_idx]  = min(sum((g - LM{i}(j, :)) .^ 2, 2));
        
        % refine grid around initial guess min_idxs
        omega_loc = [p(min_idx, 1) - 5, p(min_idx, 1) + 5, p(min_idx, 2) - 5, p(min_idx, 2) + 5];
        [xx_loc, yy_loc] = cell_centered_grid(omega_loc, [500, 500]);
        p_loc = [xx_loc(:), yy_loc(:)];
        uStar_loc(:, 1) = bilinear_interpolation(reshape(uStar(:, 1, i), [m, n]), [1, 1], p_loc);
        uStar_loc(:, 2) = bilinear_interpolation(reshape(uStar(:, 2, i), [m, n]), [1, 1], p_loc);
        g_loc = p_loc + uStar_loc;
        [~, min_idx_loc] = min(sum((g_loc - LM{i}(j, :)) .^ 2, 2));
        LM_transformed{i}(j, :) = p_loc(min_idx_loc, :);
        
    end
    
end

%
exp_save('data');
exp_end();

% input, output and low rank components in comparison
figure;
colormap gray(256);
while true
    for i = 1 : k
        subplot(1, 2, 1);
        imshow(img{i}, [], 'InitialMagnification', 'fit');
        hold on
        scatter(LM{i}(:, 2), LM{i}(:, 1));
        hold off
        title(sprintf('input T_{%d}', i));
        subplot(1, 2, 2);
        imshow(img_u{i}, [], 'InitialMagnification', 'fit');
        hold on
        scatter(LM_transformed{i}(:, 2), LM_transformed{i}(:, 1));
        hold off
        title(sprintf('output T_{%d}(u_{%d})', i, i));
        waitforbuttonpress;
    end
end