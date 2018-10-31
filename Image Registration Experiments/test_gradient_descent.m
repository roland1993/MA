%% initialization
clear all, close all, clc;

% rosenbrock function as test case
f = @(x) rosenbrock(x);
x0 = [-1.5; 1.5];
tol = 1e-3;
maxIter = 1000;

[x_star, ~, x_history] = gradient_descent(f, x0, tol, maxIter);

%% display results
figure;

% contour plot of target function
h = @(x1, x2) (1 - x1) .^ 2 + 100 * (x2 - x1 .^ 2) .^ 2;
[xx, yy] = meshgrid(linspace(-3, 3), linspace(0, 3));
zz = log(h(xx, yy));
contour(xx, yy, zz, 20);
colorbar;
axis equal;
hold on;

% plot iterates of x
plot(x_history(1, :), x_history(2, :), 'k-o');
xlabel('---x1-->');
ylabel('---x2-->');