function [dx, dy] = image_gradient(X, h)
% IN:
%   X ~ m x n       image
%   h ~ 2 x 1       grid width
% OUT:
%   dx ~ m x n      image gradient in x-direction
%   dx ~ m x n      image gradient in y-direction

[m, n] = size(X);

% use forward differences to estimate dX/dx and dX/dy
X_hor_shift = [X(:, 2 : n), zeros(m, 1)];
dx = (X_hor_shift - X) / h(1);

X_ver_shift = [zeros(1, n); X(1 : (m - 1), :)];
dy = (X_ver_shift - X) / h(2);

end