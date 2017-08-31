function [train_data, test_data, train_lbl, test_lbl] = mean_teager_features(p, train_trials, test_trials, train_labels, test_labels)

	function out = process(input)
		[channels, samples, trials] = size(input);
		
		out = zeros(1, channels, trials);
		for i=1:trials
			trial_samples = input(:,:,i);
			for j=1:channels
				row = trial_samples(j,:);
				out(1, j, i) = mean(teager(row));
			end
		end
	end
	
	train_data = process(train_trials);
	test_data = process(test_trials);
	
	% same
	train_lbl = train_labels;
	test_lbl = test_labels;
end


