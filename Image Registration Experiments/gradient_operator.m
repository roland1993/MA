function [GXtGX, GYtGY] = gradient_operator(s, h)
% IN:
%   s   ~ m x n                 grid size (assumed to be cell centered)
%   h   ~ 2 x 1                 grid width
% OUT:
%   GXtGX ~ (m*n) x (m*n)       u' * Gx' * Gx * u implements the sum of
%                                   averaged squared finite x-differences
%                                   (evaluated over x-staggered grid)
%   GYtGY ~ (m*n) x (m*n)       u' * Gy' * Gy * u implements the sum of
%                                   averaged squared finite y-differences
%                                   (evaluated over y-staggered grid)

m = s(1);   n = s(2);

% 1. finite x-differences (evaluated over x-staggered grid)
e = ones(n, 1);
D = (1 / h(1)) * spdiags([-e, e], -1 : 0, n + 1, n);
D(1, 1) = 0;    D(end, end) = 0;        % Neumann boundary condition
Dx = kron(D, speye(m));

% 2. finite y-differences (evaluated over y-staggered grid)
e = ones(m, 1);
D = (1 / h(2)) * spdiags([e, -e], -1 : 0, m + 1, m);
D(1, 1) = 0;    D(end, end) = 0;        % Neumann boundary condition
Dy = kron(speye(n), D);

% 3. averaging in x-direction
ax = [1/2; ones(n - 1, 1); 1/2];
Ax = spdiags(kron(ax, ones(m, 1)), 0, (n + 1) * m, (n + 1) * m);

% 4. averaging in y-direction
ay = [1/2; ones(m - 1, 1); 1/2];
Ay = spdiags(kron(ones(n, 1), ay), 0, n * (m + 1), n * (m + 1));

% combine to form output
GXtGX = Dx' * Ax * Dx;
GYtGY = Dy' * Ay * Dy;

end