function [img_u, dimg_u] = evaluate_displacement(img, h, u)
% IN:
%   img     ~ m x n             image
%   h       ~ 2 x 1             grid width
%   u       ~ (m*n) x 2         displacement field [u_x, u_y]
% OUT:
%   img_u   ~ m x n             interpolation of img over displaced grid
%   dimg_u  ~ (m*n) x (m*n*2)   gradient of interpol(img)

[m, n] = size(img);

% get cell centered grid
[x, y] = cell_centered_grid([m, n], h);
p = [x(:), y(:)];

% displace grid by u
p_displaced = p + u;

% interpolate X over resulting grid
if nargout == 1
    img_u = bilinear_interpolation(img, h, p_displaced);
elseif nargout == 2
    [img_u, dimg_u] = bilinear_interpolation(img, h, p_displaced);
    dimg_u = [spdiags(dimg_u(:, 1), 0, m*n, m*n), ...
        spdiags(dimg_u(:, 2), 0, m*n, m*n)];
end
img_u = reshape(img_u, [m, n]);

end