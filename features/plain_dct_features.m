% compute DCT map for single trial
% num_images = num_trials, raw DCT coefficients are used for each generated image
function [train_data, test_data, train_lbl, test_lbl] = plain_dct_features(p, train_trials, test_trials, train_labels, test_labels)

    fprintf('Computing plain DCT map signal features:\n');
    
    fact = 0.5; % percentage of DCT coefficients to leave
	
	[s,h,d] = size(train_trials);
	fprintf('  train data: %d x %d x %d, train labels: %d\n', s,h,d, length(train_labels));
	[s,h,d] = size(test_trials);
    fprintf('  test data: %d x %d x %d, test labels: %d\n', s,h,d, length(test_labels));
    
	function [to,labels] = compute(from, labels)
		[channels, samples, trials] = size(from);
		
		to = zeros(p.cnn_img_size(1), p.cnn_img_size(2), trials);
        
		for i=1:trials
			raw_eeg = from(:, :, i);
            temp = raw_eeg;
            % compress
            dct_coef = dct2(raw_eeg(:));
            dct_coef(floor(end*fact):end) = 0;
            result = idct2(dct_coef);
            
            temp = reshape(result, size(raw_eeg));
            temp = log(temp.^2);
     
            %keyboard
            to(:, :, i) = imresize(temp, p.cnn_img_size); % no nearest - add more smoothing/filtering
		end
		
		[s,h,d] = size(to);
		fprintf('  output data: %d x %d x %d\n', s,h,d);
	end
    
    [train_data, train_lbl] = compute(train_trials, train_labels);
    [test_data, test_lbl] = compute(test_trials, test_labels);
end