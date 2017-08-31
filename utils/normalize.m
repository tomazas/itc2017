% normalize signal 
function ret = normalize(s,mode)
    if nargin < 2
        mode = 0;
    end
    
    if mode == 0
        % 0 mean and unit variance
        mu = mean(s(:));
        sigma = sqrt(var(s(:)));
    else
        % 0 mean, norm by max amplitude
        mu = mean(s(:));
        sigma = max(abs(s(:)));
    end
    ret = (s - ones(size(s))*mu) ./ sigma;
end