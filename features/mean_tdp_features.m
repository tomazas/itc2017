function [train_data, test_data, train_lbl, test_lbl] = mean_tdp_features(p, train_trials, test_trials, train_labels, test_labels)
	
	function [f] = tdp_extract(s) % helper function
		d = 5;
		u = 0.015;
		params.feat.subtype = 'log-power';
		subtype = 'log-power';
		
		[ff, gg] = tdp(s, d, u);
		if (strcmp(subtype, 'log-power'))
			f = ff;
		elseif strcmp(subtype, 'log-amplitude')
			f = gg;
		elseif strcmp(subtype, 'log-power+log-amplitude')
			f = [ff, gg];
		end
	end

	function out = process(input)
	    [channels, samples, trials] = size(input);

		out = zeros(1, channels, trials);
		for i=1:trials
			trial_samples = input(:,:,i);
			for j=1:channels
				row = trial_samples(j,:);
				feats = tdp_extract(row);

				feats(~isfinite(feats)) = 0; % remove NaN
				out(1, j, i) = mean(feats);
			end
		end
	end
	
	train_data = process(train_trials);
    test_data = process(test_trials);
	
	% same
	train_lbl = train_labels;
	test_lbl = test_labels;
end


