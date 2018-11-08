function [f, df, d2f] = ...
    objective_function(dist_fctn, reg_fctn, lambda, T, R, h, u)
% IN:
%   dist_fctn   ~ function handle       distance measure
%   reg_fctn    ~ function handle       regularizer
%   lambda      ~ 1 x 1                 regularization strength parameter
%   T           ~ m x n                 template image
%   R           ~ m x n                 reference image
%   h           ~ 2 x 1                 grid width
%   u           ~ (m*n*2) x 1           displacement field
% OUT:
%   f           ~ 1 x 1                 value of objective function at u
%   df          ~ (m*n*2) x 1           gradient df/du
%   d2f         ~ (m*n*2) x (m*n*2)     (approx.) Hessian d2f/du^2

% fetch [m, n] for regularizer input
s = size(R);

% reshape u to ~ (m*n) x 2
u = reshape(u, [prod(s), 2]);

% compute distance measure and regularizer energy
[f_dist, df_dist, d2f_dist] = dist_fctn(T, R, h, u);
[f_reg, df_reg, d2f_reg] = reg_fctn(u, s, h);

% combine to form output
f = f_dist + lambda * f_reg;
df = df_dist + lambda * df_reg;
d2f = d2f_dist + lambda * d2f_reg;

end