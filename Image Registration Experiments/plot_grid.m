function plot_grid(g, s, f)
% IN:
%   g ~ m x n x 2           grid to plot
%   s ~ 1 x 1               spacing for grid plotting
%   f ~ figure handle       figure to plot in

% set active figure
if nargin < 3
    f = gcf;
end
if nargin < 2
    s = 1;
end
figure(f);
hold on;

% fetch grid size
[m, n, ~] = size(g);

% plot horizontal grid lines
for i = 1 : s : m
    plot(g(i, 1 : s : end, 1), g(i, 1 : s : end, 2), ...
        'm-o', 'LineWidth', 0.1, 'MarkerSize', 4);
end

% plot vertical grid lines
for j = 1 : s : n
    plot(g(1 : s : end, j, 1), g(1 : s : end, j, 2), ...
        'm-o', 'LineWidth', 0.1, 'MarkerSize', 4);
end

hold off;

end