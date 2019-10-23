function LM_acc = landmark_accuracy(LM)
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
%   LM      ~ numLM x 2     landmark coordinates
% OUT:
%   LM_acc  ~ numLM x 1     mean dist of each lm to respective mean-lm
%--------------------------------------------------------------------------


k = numel(LM);
y = zeros([size(LM{1}), k]);

for i = 1 : k
    y(:, :, i) = LM{i};
end
y_bar = mean(y, 3);

LM_acc = sum( sqrt(sum((y - y_bar) .^ 2, 2)), 3) / k;

end