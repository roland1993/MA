% clean-up
clear all, close all, clc;

% problem size
m = 30;
n = 20;

% randomize operator and offset
K = randi(3, m, n) - 2;
g = randi(11, n, 1) - 6;

% initialize starting point
x0 = 10 * randn(n, 1);
y0 = 10 * randn(m, 1);

% define primal/dual step sizes with    tau * sigma * L^2 < 1
[~, D] = svd(K, 'econ');
L = D(1, 1);
tau = sqrt((1 - 1e-4) / L ^ 2);
sigma = tau;

% function handles for F and G
lambda = 10;
F = @(y, c_flag) test_F(y, sigma, c_flag);
G = @(x, c_flag) test_G(x, g, lambda, tau, c_flag);

% perform optimization
theta = 1;
[x_star, y_star, primal_history, dual_history] = ...
    chambolle_pock(F, G, K, x0, y0, theta, tau, sigma);

% plot results
figure;
plot(1 : numel(primal_history), primal_history, ...
    1 : numel(primal_history), dual_history);
grid on;    axis tight;
legend('primal energy', 'dual energy');     xlabel('#iter');