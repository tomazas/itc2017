% compute EEG energy map by squaring the EEG signal, i.e. x^2 for each channel
% num_images = num_trials, raw DCT coefficients are used for each generated image
function [train_data, test_data, train_lbl, test_lbl] = energy_map_features(p, train_trials, test_trials, train_labels, test_labels)

    fprintf('Computing plain energy map signal features:\n');

	[s,h,d] = size(train_trials);
	fprintf('  train data: %d x %d x %d, train labels: %d\n', s,h,d, length(train_labels));
	[s,h,d] = size(test_trials);
    fprintf('  test data: %d x %d x %d, test labels: %d\n', s,h,d, length(test_labels));
    
	function [to,labels] = compute(from, labels)
		[channels, samples, trials] = size(from);
		
		to = zeros(p.cnn_img_size(1), p.cnn_img_size(2), trials);
        
		for i=1:trials
			raw_eeg = from(:, :, i);
            temp = log(raw_eeg.^2);
            to(:, :, i) = imresize(temp, p.cnn_img_size); % no nearest - add more smoothing/filtering
		end
		
		[s,h,d] = size(to);
		fprintf('  output data: %d x %d x %d\n', s,h,d);
	end
    
    [train_data, train_lbl] = compute(train_trials, train_labels);
    [test_data, test_lbl] = compute(test_trials, test_labels);
end