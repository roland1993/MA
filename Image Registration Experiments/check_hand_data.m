function check_hand_data
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

% ensure that hands-R.jpg and hands-T.jpg from FAIR.m are present
%   -> if not, download from web and save in current working directory
if ~isfile('hands-R.jpg') || ~isfile('hands-T.jpg')
    fprintf('\n%s WARNING %s\n', ...
        repmat('-', [1, 35]), repmat('-', [1, 36]));
    fprintf('\thands-R.jpg OR hands-T.jpg WAS NOT FOUND!');
    fprintf(...
        '\n\tDOWNLOADING DATA FROM https://github.com/C4IR/FAIR.m\n');
    websave('hands-R.jpg', ['https://raw.githubusercontent.com/', ...
        'C4IR/FAIR.m/master/kernel/data/hands-R.jpg']);
    websave('hands-T.jpg', ['https://raw.githubusercontent.com/', ...
        'C4IR/FAIR.m/master/kernel/data/hands-T.jpg']);
    fprintf('%s\n\n', repmat('-', [1, 80]));    
end

end