function [X_u, dX_u] = evaluate_displacement(X, h, u)
% IN:
%   X       ~ m x n             image
%   h       ~ 2 x 1             grid width
%   u       ~ (m*n) x 2         displacement field [u_x, u_y]
% OUT:
%   X_u     ~ m x n             interpolation of X over grid displaced by u
%   dX_u    ~ (m*n) x (m*n*2)   gradient of interpol(X) over displaced grid

[m, n] = size(X);

% get cell centered grid
[x, y] = cell_centered_grid([m, n], h);
p = [x(:), y(:)];

% displace grid by u
p_displaced = p + u;

% interpolate X over resulting grid
if nargout == 1
    X_u = bilinear_interpolation(X, h, p_displaced);
elseif nargout == 2
    [X_u, dX_u] = bilinear_interpolation(X, h, p_displaced);
    dX_u = [spdiags(dX_u(:, 1), 0, m*n, m*n), ...
        spdiags(dX_u(:, 2), 0, m*n, m*n)];
end
X_u = reshape(X_u, [m, n]);

end