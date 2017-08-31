function [y, Q] = preprocess(input, labels, Q)
    % remove mean
    m = mean(input, 3);
    y = input - m;
	Q = [];
end