function [res1, res2, res3] = zero_function(x, conjugate_flag)
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

if ~conjugate_flag
    res1 = 0;
    res2 = 0;
    if nargout == 3, res3 = x; end
else
    res1 = 0;
    res2 = max(abs(x));
    if nargout == 3, res3 = 0 * x; end
end

end