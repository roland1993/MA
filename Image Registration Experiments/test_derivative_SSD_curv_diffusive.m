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

%% initialization
clear all, close all, clc;

R = double(imread('rect1.png'));
T = double(imread('rect2.png'));
h = [1, 1];
s = size(R);

% randomize displacement field
u0 = randn(2 * prod(s), 1);

%% test SSD derivative
SSD_handle = @(u) SSD(T, R, h, u);
derivative_test(SSD_handle, u0, 1);

%% test curvature_energy derivative
curv_handle = @(u) curvature_energy(u, s, h);
derivative_test(curv_handle, u0);

%% test diffusive_energy derivative
diff_handle = @(u) diffusive_energy(u, s, h);
derivative_test(diff_handle, u0);