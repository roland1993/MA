function [res1, res2, res3] = pointwise_2x2_nn(M, mu, tau, conjugate_flag)

% evaluate mu * sum_i || v_i ||_*
if ~conjugate_flag
    
    % evaluate prox?
    if nargout == 3
        
        res1 = [];
        res2 = [];
        
        M = reshape(M, [], 4);
        [U, Sigma, V] = svd_2x2(M);
        
        res3 = zeros(size(M));
        Sigma_threshold = max(Sigma - mu * tau, 0);
        res3(:, 1) = Sigma_threshold(:, 1) .* U(:, 1) .* V(:, 1) + ...
            Sigma_threshold(:, 2) .* U(:, 3) .* V(:, 3);
        res3(:, 2) = Sigma_threshold(:, 1) .* U(:, 2) .* V(:, 1) + ...
            Sigma_threshold(:, 2) .* U(:, 4) .* V(:, 3);
        res3(:, 3) = Sigma_threshold(:, 1) .* U(:, 1) .* V(:, 2) + ...
            Sigma_threshold(:, 2) .* U(:, 3) .* V(:, 4);
        res3(:, 4) = Sigma_threshold(:, 1) .* U(:, 2) .* V(:, 2) + ...
            Sigma_threshold(:, 2) .* U(:, 4) .* V(:, 4);
        res3 = res3(:);
        
    else
        
        M = reshape(M, [], 4);
        [~, Sigma, ~] = svd_2x2(M);
        
        res1 = mu * sum(Sigma(:));
        res2 = 0;
        res3 = [];
        
    end
    
else
    
    % evaluate prox?
    if nargout == 3
        
        res1 = [];
        res2 = [];
        
        [~, ~, conj_prox] = ...
            pointwise_2x2_nn(M(:) / tau, mu, 1 / tau, ~conjugate_flag);
        res3 = M(:) - tau * conj_prox;
        
    else
        
        M = reshape(M, [], 4);
        [~, Sigma, ~] = svd_2x2(M);
        
        res1 = 0;
        res2 = max([(Sigma(:, 1) - mu) / mu; 0]);
        res3 = [];
        
    end
    
end

end