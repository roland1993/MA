function [res1, res2, res3] = TV_denoise(v, sigma, conjugate_flag)
% IN:
%       v               ~ m*n*2 x 1         gradient image to regularize
%       sigma           ~ 1 x 1             step length for prox operator
%       conjugate_flag  ~ logical           evaluate TV or TV* at v?
% OUT:
%   IF conjugate_flag:
%       res1            ~ 1 x 1             convex conjugate value TV*(v)
%       res2            ~ 1 x 1             constraint violation measure
%       res3            ~ m*n*2 x 1         prox-step of TV* for v
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1             function value TV(v)
%       res3            ~ 1 x 1             constraint violation measure
%       res2            ~ m*n*2 x 1         prox-step of TV for v

% by default: evaluate TV instead of its conjugate
if nargin < 3, conjugate_flag = false; end

% initialize constraint measure with 0
res2 = 0;

% reshape v to ~ m*n x 2
v = reshape(v, [], 2);
v_squared = v .^ 2;

% compute pointwise 2-norm of v
norm_v = sqrt(sum(v_squared, 2));

if ~conjugate_flag
    % EITHER ~> evaluate TV and Prox_TV at v
    
    % TV(v) = sum_ij ||v_ij||_2
    res1 = sum(norm_v);
    
    if nargout == 3
        
        % compute prox-step for F = TV with Moreau's identity
        %   [(id + sigma * dF)^(-1)](v) =
        %       v - sigma * [(id + (1 / sigma) * dF*)^(-1)](v / sigma)
        [~, ~, conjugate_prox] = TV(v / sigma, 1 / sigma, true);
        res3 = v - sigma * conjugate_prox;
        
    end
    
else
    % OR ~> evaluate convex conjugate TV* and Prox_TV* at v
    
    % conxex conjugate TV*(v) is indicator of {v : max_ij ||v_ij||_2 <= 1}
    if max(norm_v) > 1
        res2 = max(norm_v) - 1;
    end
    res1 = 0;
    
    if nargout == 3
        
        % prox-step on v for F* = TV* ~> pointwise reprojection
        n = max(norm_v, 1);
        res3 = v ./ n;
        res3 = res3(:);
        
    end
    
end

end