%% initialization
clear all, close all, clc;

% rosenbrock function as test case
f = @(x) rosenbrock(x);
x0 = [-1.5; 1.5];
[x_star, x_history] = newton_scheme(f, x0);

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
     ' + iterates x^k of newton scheme']);