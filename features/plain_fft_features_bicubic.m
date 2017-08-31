% compute FFT energy of each channel
% num_images = num_trials, raw FFT power is used for each generated image
function [train_data, test_data, train_lbl, test_lbl] = plain_fft_features_bicubic(p, train_trials, test_trials, train_labels, test_labels)

    fprintf('Computing plain FFT power signal features:\n');
	
	[s,h,d] = size(train_trials);
	fprintf('  train data: %d x %d x %d, train labels: %d\n', s,h,d, length(train_labels));
	[s,h,d] = size(test_trials);
    fprintf('  test data: %d x %d x %d, test labels: %d\n', s,h,d, length(test_labels));
    
	function [to,labels] = compute(from, labels)
		[channels, samples, trials] = size(from);
		
        N_fft = 1024;
        usable_fft = N_fft / 2; % only half is usable
        
        % compute usable/interesting FFT data in frequency region (ROI)
        freq_from = 0;
        freq_to = 30;
        
        max_freq = p.fs/2; % FFT contains frequencies in range [0; Fs/2]
        start_idx = max(1,floor(freq_from * usable_fft / max_freq));
        end_idx = floor(freq_to * usable_fft / max_freq);
        
        nfft_samples = end_idx-start_idx+1;
		to = zeros(p.cnn_img_size(1), p.cnn_img_size(2), trials);
        temp_im = zeros(channels, nfft_samples);

		for i=1:trials
            for j=1:channels
                raw_eeg = from(j, :, i);
                
                % compute FFT power
                Y = fft(raw_eeg, N_fft);
                Y(1) = 0; % remove the sum of all coefficients - DC component
                Y = Y(1:usable_fft); % only half of the FFT data is usable
                
                power = abs(Y(start_idx:end_idx)); % trim to interesting range   
                temp_im(j, :) = power;
            end
            
            %ensure required img size reached
            to(:, :, i) = imresize(temp_im, p.cnn_img_size, 'bicubic');
        end

		[s,h,d] = size(to);
		fprintf('  output data: %d x %d x %d\n', s,h,d);
	end
    
    [train_data, train_lbl] = compute(train_trials, train_labels);
    [test_data, test_lbl] = compute(test_trials, test_labels);
end