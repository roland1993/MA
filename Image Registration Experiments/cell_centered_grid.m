function [x, y] = cell_centered_grid(s, h)
% IN:
%   s ~ 2 x 1       image size [m, n]
%   h ~ 2 x 1       grid width [h_x, h_y]
% OUT:
%   x ~ m x n       x-coordinates of grid points
%   y ~ m x n       y-coordinates of grid points

m = s(1);
n = s(2);

x = repmat((h(1) / 2) : h(1) : (n - (1 / 2)) * h(1), [m, 1]);
y = repmat((m - (1 / 2)) * h(2) : (- h(2)) : (h(2) / 2), [n, 1])';

end