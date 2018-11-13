function ML = multi_level(img, h)
% IN:
%   img ~ m x n         input image (high-res)
%   h   ~ 2 x 1         grid width
% OUT:
%   ML  ~ cell          multi-level representation of img with attributes
%                           img:    downsampled version 
%                           s:      current resolution
%                           h:      current grid width

[m, n] = size(img);

% filter kernel for averaging over 2x2 regions
k = 0.25 * ones(2);

% reduce resolution by factor 2 as long as (m >=16) && (n >= 16)
num_levels = min(floor(log2([m, n] / 16))) + 1;

% initialize output as cell array
ML = cell(num_levels, 1);

% highest resolution level equals input data
ML{num_levels}.img = img;
ML{num_levels}.s = [m, n];
ML{num_levels}.h = h;

% build all levels
for i = (num_levels - 1) : (-1) : 1
    
    % reducing resolution by factor 2 => increasing grid width by factor 2
    ML{i}.h = 2 * ML{i + 1}.h;
    
    % average higher resolution image over 2x2 regions
    img_average = conv2(ML{i + 1}.img, k, 'same');
    
    % take every second pixel in each dimension for low-res image
    ML{i}.img = img_average(1 : 2 : end, 1 : 2 : end);
    ML{i}.s = size(ML{i}.img);
    
end

end