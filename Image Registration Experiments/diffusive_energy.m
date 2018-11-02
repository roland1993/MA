function [f, df] = diffusive_energy(u, s, h)
% IN:
%   u   ~ (m*n) x 2         displacement field to regularize
%   s   ~ 2 x 1             s = [m, n]
%   h   ~ 2 x 1             grid width
% OUT:
%   f   ~ 1 x 1             diffusive energy of u
%   df  ~ (m*n*2) x 1       derivative of diffusive energy (by u)

m = s(1);
n = s(2);

% get gradient operators for x and y-direction
[~, ~, G_x, G_y] = image_gradient(zeros(m, n), h);

% apply gradient operators to u_x and u_y
u_x_x = G_x * u(:, 1);
u_x_y = G_y * u(:, 1);
u_y_x = G_x * u(:, 2);
u_y_y = G_y * u(:, 2);

% compute diffusiv energy for u
f = 0.5 * (u_x_x' * u_x_x + u_x_y' * u_x_y + ...
    u_y_x' * u_y_x + u_y_y' * u_y_y);

% compute df/du
G = kron(speye(2), [G_x; G_y]);
df = G' * [u_x_x; u_x_y; u_y_x; u_y_y];

end