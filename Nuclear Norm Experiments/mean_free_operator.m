function B = mean_free_operator(m, n, k)
% IN:
%   m ~ 1 x 1                   number of image rows
%   n ~ 1 x 1                   number of image columns
%   k ~ 1 x 1                   number of images
% OUT:
%   B ~ sparse(m*n*k x m*n*k)   mean subtraction operator

B = kron(speye(k) - ones(k) / k, speye(m * n));

end