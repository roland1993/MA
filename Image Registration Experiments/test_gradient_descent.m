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

% rosenbrock function as test case
f = @(x) rosenbrock(x);
x0 = [-1.5; 1.5];
tol1 = 1e-3;
maxIter = 1000;

[x_star, x_history] = gradient_descent(f, x0, tol1, maxIter);

%% display results
figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% contour plot of target function
h = @(x1, x2) (1 - x1) .^ 2 + 100 * (x2 - x1 .^ 2) .^ 2;
[xx, yy] = meshgrid(linspace(-3, 3, 500), linspace(0, 3, 500));
zz = log(h(xx, yy));
contour(xx, yy, zz, 15);
colorbar;
axis equal;
hold on;

% plot iterates of x
plot(x_history(1, :), x_history(2, :), 'k-o', 'MarkerSize', 4);
xlabel('---x1-->');
ylabel('---x2-->');
title(['logarithmic plot of rosenbrock function', ...
     ' + iterates x^k of gradient descent']);