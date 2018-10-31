function plot_grid(g, f)
% IN:
%   g ~ m x n x 2           grid to plot
%   f ~ figure handle       figure to plot in

% set active figure
if nargin < 2
    f = gcf;
end
figure(f);
hold on;

% fetch grid size
[m, n, ~] = size(g);

% plot horizontal grid lines
for i = 1 : 2 : m
    plot(g(i, 1 : 2 : end, 1), g(i, 1 : 2 : end, 2), ...
        'm-o', 'LineWidth', 0.1, 'MarkerSize', 4);
end

% plot vertical grid lines
for j = 1 : 2 : n
    plot(g(1 : 2 : end, j, 1), g(1 : 2 : end, j, 2), ...
        'm-o', 'LineWidth', 0.1, 'MarkerSize', 4);
end

hold off;

end