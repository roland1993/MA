function [f, df, d2f] = diffusive_energy(u, s, h)
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
%   u   ~ (m*n) x 2             displacement field to regularize
%   s   ~ 2 x 1                 s = [m, n]
%   h   ~ 2 x 1                 grid width
% OUT:
%   f   ~ 1 x 1                 diffusive energy of u
%   df  ~ (m*n*2) x 1           gradient of diffusive energy w.r.t. u
%   d2f ~ (m*n*2) x (m*n*2)     Hessian of diffusive energy w.r.t u
%--------------------------------------------------------------------------

% make sure u has the right format...
u = reshape(u, [], 2);

persistent GtG_h;

% compute difference operator G' * G only once (for every resolution level)
if isempty(GtG_h) || size(GtG_h, 2) ~= numel(u)
    
    % get gradient operators for x and y-direction
    [GXtGX, GYtGY] = gradient_operator(s, h);
    
    % form new gradient operator to operate on u(:)
    GtG_h = prod(h) * kron(speye(2), (GXtGX + GYtGY));
    
end

d2f = GtG_h;                % d2f = prod(h) * G' * G
df = GtG_h * u(:);          % df  = prod(h) * G' * G * u
f = 0.5 * u(:)' * df;       % f   = 0.5 * prod(h) * u' * G' * G * u

end