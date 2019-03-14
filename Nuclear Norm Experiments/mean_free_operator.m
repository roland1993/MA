function B = mean_free_operator(m, n, k)
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
%   m ~ 1 x 1                   number of image rows
%   n ~ 1 x 1                   number of image columns
%   k ~ 1 x 1                   number of images
% OUT:
%   B ~ sparse(m*n*k x m*n*k)   mean subtraction operator
%--------------------------------------------------------------------------

B = kron(speye(k) - ones(k) / k, speye(m * n));

end