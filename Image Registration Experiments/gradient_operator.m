function [G_x, G_y] = gradient_operator(s, h)
% IN:
%   s   ~ m x n             grid size (assumed to be cell centered)
%   h   ~ 2 x 1             grid width
% OUT:
%   G_x ~ (m*n) x (m*n)     x-gradient operator for X in column major
%   G_y ~ (m*n) x (m*n)     y-gradient operator for X in column major

m = s(1);   n = s(2);

% switch from cell centered to staggered grid (x-direction / y-direction)
% i) x-direction
e = ones(n, 1);
D = 0.5 * spdiags([e, e], -1 : 0, n + 1, n);
S_x = kron(D, speye(m));
% ii) y-direction
e = ones(m, 1);
D = 0.5 * spdiags([e, e], -1 : 0, m + 1, m);
S_y = kron(speye(n), D);

% define differential operators
% i) x-derivative
e = ones(n, 1);
D = (1 / h(1)) * spdiags([-e, e], 0 : 1, n, n + 1);
D_x = kron(D, speye(m));
% ii) y-derivative
e = ones(m, 1);
D = (1 / h(2)) * spdiags([e, - e], 0 : 1, m, m + 1);
D_y = kron(speye(n), D);

% calculate x-derivative and y-derivative of input image
G_x = D_x * S_x;
G_y = D_y * S_y;

end