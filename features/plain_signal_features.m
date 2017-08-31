% select bare/raw EEG signal values as features from all EEG channels
% num_images = num_trials
function [train_data, test_data, train_lbl, test_lbl] = plain_signal_features(p, train_trials, test_trials, train_labels, test_labels)

    fprintf('Computing plain signal features:\n');
	
	[s,h,d] = size(train_trials);
	fprintf('  train data: %d x %d x %d, train labels: %d\n', s,h,d, length(train_labels));
	[s,h,d] = size(test_trials);
    fprintf('  test data: %d x %d x %d, test labels: %d\n', s,h,d, length(test_labels));
    
	function [to,labels] = compute(from, labels)
		[channels, samples, trials] = size(from);
		
		to = zeros(p.cnn_img_size(1), p.cnn_img_size(2), trials);
        
		for i=1:trials
			raw_eeg = from(:, :, i);
            to(:, :, i) = imresize(raw_eeg, p.cnn_img_size, 'nearest');
		end
		
		[s,h,d] = size(to);
		fprintf('  output data: %d x %d x %d\n', s,h,d);
	end
    
    [train_data, train_lbl] = compute(train_trials, train_labels);
    [test_data, test_lbl] = compute(test_trials, test_labels);
end