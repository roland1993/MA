function [x, y] = staggered_grid(s, h, dir)
% IN:
%   s   ~ 2 x 1         image size [m, n]
%   h   ~ 2 x 1         grid width [h_x, h_y]
%   dir ~ char          direction of staggered grid 'x' or 'y'
% OUT:
%   x   ~ m x (n+1)     x-coordinates of grid points (in case dir = 'x')
%       ~ (m+1) x n     x-coordinates of grid points (in case dir = 'y')
%   y   ~ m x (n+1)     y-coordinates of grid points (in case dir = 'x')
%       ~ (m+1) x n     y-coordinates of grid points (in case dir = 'y')

m = s(1);
n = s(2);

if dir == 'x'
    x = repmat(0 : h(1) : n * h(1), [m, 1]);
    y = repmat((m - (1 / 2)) * h(2) : (- h(2)) : (h(2) / 2), [n + 1, 1])';
elseif dir == 'y'
    x = repmat((h(1) / 2) : h(1) : (n - (1 / 2)) * h(1), [m + 1, 1]);
    y = repmat(m * h(2) : (-h(2)) : 0, [n, 1])';
end

end