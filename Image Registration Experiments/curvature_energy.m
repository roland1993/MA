function [f, df] = curvature_energy(u, s, h)
% IN:
%   u ~ (m*n) x 2       displacement field to regularize
%   s ~ 2 x 1           s = [m, n]
%   h ~ 2 x 1           grid width
% OUT:
%   f ~ 1 x 1           curvature energy of u
%   df ~ (m*n*2) x 1    gradient of curvature w.r.t u

m = s(1);   n = s(2);

% create discrete Laplacian operator
%   i)  2nd order x-derivative
e_x = ones(n,1);
D_xx = (1 / h(1)) ^ 2 * spdiags([e_x, -2*e_x, e_x], -1 : 1, n, n);
%   ii) 2nd order y-derivative
e_y = ones(m, 1);
D_yy = (1 / h(2)) ^ 2 * spdiags([e_y, -2*e_y, e_y], -1 : 1, m, m);
%   iii) combine to form discrete Laplacian
L = kron(D_xx, speye(m)) + kron(speye(n), D_yy);

% compute curvature energy 0.5 * [ux' * L' * L * ux + uy' * L' * L * uy]
L_u_x = L * u(:, 1);
L_u_y = L * u(:, 2);
f = 0.5 * (L_u_x' * L_u_x + L_u_y' * L_u_y);

% compute df/du as [L' 0; 0 L'] * [L 0; 0 L] * [ux; uy]
if nargout == 2
    df = [L * L_u_x; L * L_u_y];
end

end