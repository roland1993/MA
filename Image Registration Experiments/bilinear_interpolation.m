function [X_p, dX_p] = bilinear_interpolation(X, h, p)
% IN:
%   X       ~ m x n     image
%   h       ~ 2 x 1     grid width [h_x, h_y]
%   p       ~ l x 2     evaluation points
% OUT:
%   X_p     ~ l x 1     bilinear interpolation of X at points p
%   dX_p    ~ l x 2     x- and y-derivative of interpol(X) at points p

[m, n] = size(X);
l = size(p, 1);

% transform (x, y)-coordinates to (i, j)-coordinates
q = p - (h / 2);
r = q ./ h;
s = [r(:, 1), -r(:, 2) + (m - 1)];
t = [s(:, 2), s(:, 1)];
u = t + 1;

% pad X with zeros for boundary condition
Y = padarray(X, [1 1]);

% evaluate linear interpolation of X at given points p
X_p = zeros(l, 1);

% evaluate x- and y-derivative of interpol(X) if requested
if nargout == 2
    dX_p = zeros(l, 2);
end

for i = 1 : l
    
    % set interpolation value to 0 for (i, j) not in [0, m+1] x [0, n+1]
    %   <=> (x, y) not in [0, n*h1] x [0, m*h2]
    if (u(i, 1) <= 0) || (u(i, 1) >= (m + 1)) || ...
            (u(i, 2) <= 0) || (u(i, 2) >= (n + 1))
        X_p(i) = 0;
        if nargout == 2
            dX_p(i, :) = [0, 0];
        end
    else
        int = floor(u(i, :));
        dec = u(i, :) - int;
        
        X_p(i) = ...
            (1 - dec(1)) * (1 - dec(2)) * Y(int(1) + 1, int(2) + 1) + ...
            dec(1) * (1 - dec(2)) * Y(int(1) + 2, int(2) + 1) + ...
            (1 - dec(1)) * dec(2) * Y(int(1) + 1, int(2) + 2) + ...
            dec(1) * dec(2) * Y(int(1) + 2, int(2) + 2);
        
        if nargout == 2
            % x-derivative
            dX_p(i, 1) = (1 / h(1)) * ...
                (1 - dec(1)) * (-1) * Y(int(1) + 1, int(2) + 1) + ...
                dec(1) * (-1) * Y(int(1) + 2, int(2) + 1) + ...
                (1 - dec(1)) * (+1) * Y(int(1) + 1, int(2) + 2) + ...
                dec(1) * (+1) * Y(int(1) + 2, int(2) + 2);
            % y-derivative
            dX_p(i, 2) = (1 / h(2)) * ...
                (+1) * (1 - dec(2)) * Y(int(1) + 1, int(2) + 1) + ...
                (-1) * (1 - dec(2)) * Y(int(1) + 2, int(2) + 1) + ...
                (+1) * dec(2) * Y(int(1) + 1, int(2) + 2) + ...
                (-1) * dec(2) * Y(int(1) + 2, int(2) + 2);
        end
    end
    
end

end