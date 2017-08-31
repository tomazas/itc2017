% computes mean and variance of each channel and compacts such feature pairs into single value row
function [train_data, test_data, train_lbl, test_lbl] = mean_var_features(p, train_trials, test_trials, train_labels, test_labels)

    % build training set
    function out = prepare(input)
        [channels, ~, trials] = size(input); % [channels, samples, trials]
        out = zeros(1, p.cnn_img_size(2), trials);
        temp = zeros(1, channels);
        
        for i=1:trials
            trial_samples = input(:,:,i);
            
            for j=1:channels
                eeg = trial_samples(j,:);
                temp(j) = var(eeg);
            end
            
            out(1, :, i) = imresize(log(temp), [1 p.cnn_img_size(2)], 'nearest');
        end
    end

    % same
    train_lbl = train_labels;
	test_lbl = test_labels;
	
    train_data = prepare(train_trials);
    test_data = prepare(test_trials);
end


