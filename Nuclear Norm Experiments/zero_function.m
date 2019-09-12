function [res1, res2, res3] = zero_function(x, conjugate_flag)
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
%       x               ~ l x 1         input vector
%       conjugate_flag  ~ logical       evaluate conjugate?
% OUT:
%   IF conjugate_flag:
%       res1            ~ 1 x 1         0 for all x
%       res2            ~ 1 x 1         constraint violation measure
%       res3            ~ l x 1         prox of 0-fctn. = identity(x)
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         delta_{0}(x)
%       res2            ~ 1 x 1         constraint violation measure
%       res3            ~ l x 1         prox of delta{0} = 0 for all x
%--------------------------------------------------------------------------

% use GPU?
GPU = isa(x, 'gpuArray');
if GPU
    data_type = 'gpuArray';
else
    data_type = 'double';
end

if ~conjugate_flag
    res1 = zeros(1, data_type);
    res2 = zeros(1, data_type);
    if nargout == 3, res3 = x; end
else
    res1 = zeros(1, data_type);
    res2 = max(abs(x));
    if nargout == 3, res3 = zeros(size(x), data_type); end
end

end