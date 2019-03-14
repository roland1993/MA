function u_high_res = prolong_displacement(u_low_res, s_in, s_out)
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
%   u_low_res   ~ (m*n*2) x 1       grid displacement at low resolution
%   s_in        ~ 2 x 1             s_in = [m, n] resolution of input
%   s_out       ~ 2 x 1             s_out = [k, l] target resolution
% OUT:
%   u_high_res  ~ (k*l*2) x 1       grid displacement at high resolution
%--------------------------------------------------------------------------

m = s_in(1);    n = s_in(2);
k = s_out(1);   l = s_out(2);

u_low_res = reshape(u_low_res, [m, n, 2]);
u_prolonged = cat(3, ...
    kron(u_low_res(:, :, 1), ones(2)), ...
    kron(u_low_res(:, :, 2), ones(2)));

% cut out the relevant part of u_prolonged ~ (2m) x (2n) x 2
u_high_res = u_prolonged(1 : k, 1 : l, :);
u_high_res = u_high_res(:);

end