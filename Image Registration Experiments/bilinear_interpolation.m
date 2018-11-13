function [img_p, dimg_p] = bilinear_interpolation(img, h, p)
% IN:
%   X           ~ m x n     image
%   h           ~ 2 x 1     grid width [h_x, h_y]
%   p           ~ l x 2     evaluation points (in xy-coordinates)
% OUT:
%   img_p       ~ l x 1     bilinear interpolation of img at points p
%   dimg_p      ~ l x 2     x-/y-derivative of interpol(img) at points p

[m, n] = size(img);
l = size(p, 1);

% use homogeneous coordinates for p
p = [p, ones(size(p(:, 1)))];
% define world matrix to switch from xy-coordinates to ij-coordinates
W = [1/h(1), 0, 1/2; ...
    0, 1/h(2), 1/2; ...
    0, 0, 1];
% switch coordinates (and remove homogeneous component)
q = W * p';
q = q(1 : 2, :)';

% pad img for dirichlet boundary condition
img_pad = padarray(img, [1, 1]);

% evaluate bilinear interpolation of img at points p (or q respectively)
img_p = zeros(l, 1);

% evaluate x-/y-derivative of interpol(img) (if requested)
if nargout == 2
    dimg_p = zeros(l, 2);
end

for i = 1 : l
    
    % set values for (i,j) outside of [0,m+1]x[0,n+1] to 0
    if (q(i, 1) <= 0) || (q(i, 1) >= (m + 1)) || ...
            (q(i, 1) <= 0) || (q(i, 2) >= (n + 1))
        img_p(i) = 0;
        if nargout == 2, dimg_p(i, :) = [0, 0]; end
    else
        
        int = floor(q(i, :));
        dec = q(i, :) - int;
        
        img_p(i) = ...
            (1-dec(1)) * (1-dec(2)) * img_pad(int(1) + 1, int(2) + 1) + ...
            (1-dec(1)) * dec(2) * img_pad(int(1) + 1, int(2) + 2) + ...
            dec(1) * (1-dec(2)) * img_pad(int(1) + 2, int(2) + 1) + ...
            dec(1) * dec(2) * img_pad(int(1) + 2, int(2) + 2);
        
        if nargout == 2
            % x-derivative
            dimg_p(i, 1) = (1 / h(1)) * (...
                (-1) * (1-dec(2)) * img_pad(int(1) + 1, int(2) + 1) + ...
                (-1) * dec(2) * img_pad(int(1) + 1, int(2) + 2) + ...
                (+1) * (1-dec(2)) * img_pad(int(1) + 2, int(2) + 1) + ...
                (+1) * dec(2) * img_pad(int(1) + 2, int(2) + 2));
            % y-derivative
            dimg_p(i, 2) = (1 / h(2)) * (...
                (-1) * (1-dec(1)) * img_pad(int(1) + 1, int(2) + 1) + ...
                (+1) * (1-dec(1)) * img_pad(int(1) + 1, int(2) + 2) + ...
                (-1) * dec(1) * img_pad(int(1) + 2, int(2) + 1) + ...
                (+1) * dec(1) * img_pad(int(1) + 2, int(2) + 2));
        end
        
    end
    
end

end