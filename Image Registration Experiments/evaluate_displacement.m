function [X_u, dX_u] = evaluate_displacement(X, h, u)
% IN:
%   X ~ m x n           image
%   h ~ 2 x 1           grid width
%   u ~ (m*n) x 2       displacement field [u_x, u_y]
% OUT:
%   X_u ~ m x n         interpolation of X over grid displaced by u
%   dX_u ~ m x n x 2    interpolated gradient of X over displaced grid
    
[m, n] = size(X);

% get cell centered grid
[x, y] = cell_centered_grid([m, n], h);
p = [x(:), y(:)];

% displace grid by u
p_displaced = p + u;

% interpolate X over resulting grid
X_u = bilinear_interpolation(X, h, p_displaced);
X_u = reshape(X_u, [m, n]);

% return gradient of interpolated image (if requested)
if (nargout == 2)
    
    % compute image gradient of X only once, save as persistent variables
    persistent dx;  persistent dy;
    if isempty(dx)
        [dx, dy] = image_gradient(X, h);
    end
    
    % interpolate dX_u/dx and dX_u/dy from dX/dx and dX_/dy
    dX_u_x = bilinear_interpolation(dx, h, p_displaced);
    dX_u_y = bilinear_interpolation(dy, h, p_displaced);
    
    dX_u = zeros(m, n, 2);
    dX_u(:, :, 1) = reshape(dX_u_x, [m, n]);
    dX_u(:, :, 2) = reshape(dX_u_y, [m, n]);
    
end

end