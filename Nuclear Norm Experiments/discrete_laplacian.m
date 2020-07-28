function L = discrete_laplacian(m, n, h_grid, k)
%--------------------------------------------------------------------------
% This file is part of my master's thesis entitled
%           'Low rank- and sparsity-based image registration'
% For the whole project see
%           https://github.com/roland1993/MA
% If you have questions contact me at
%           roland.haase [at] student.uni-luebeck [dot] de
% Source code is provided under the
%           MIT Open Source License
%--------------------------------------------------------------------------
% IN:
%   m       ~ 1 x 1                         number of rows
%   n       ~ 1 x 1                         number of columns
%   h_grid  ~ 1 x 2                         grid spacing
%   k       ~ 1 x 1                         number of grids
% OUT:
%   D       ~ sparse(2*k*m*n x 2*k*m*n)     discrete laplacian per grid
%--------------------------------------------------------------------------

% x-derivative
e_x = ones(m, 1);
D_xx = (1 / h_grid(1)) * spdiags([e_x, (-2) * e_x, e_x], -1 : 1, m, m);
D_xx(1, 1) = (-1) / h_grid(1);
D_xx(end, end) = (-1) / h_grid(1);

% y-derivative
e_y = ones(n, 1);
D_yy = (1 / h_grid(2)) * spdiags([e_y, (-2) * e_y, e_y], -1 : 1, n, n);
D_yy(1, 1) = (-1) / h_grid(2);
D_yy(end, end) = (-1) / h_grid(2);

% construct laplacian, extend to 2*k grids (one per coordinate direction)
L = kron(speye(2 * k), kron(speye(n), D_xx) + kron(D_yy, speye(m)));

end