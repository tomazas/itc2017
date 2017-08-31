% principal component analysis
function [train_data, test_data, train_lbl, test_lbl] = pca_features(p, train_trials, test_trials, train_labels, test_labels)
   
    disp('Computing features');

	function [features] = compute(trial_data)
	    [channels, samples, trials] = size(trial_data);
		features = zeros(1, channels, trials);
		for i=1:trials
			trial_samples = trial_data(:,:,i);
            
            level = 5;
            wname = 'sym4';
            npc = 'kais';

            [x_sim, qual, npc] = wmspca(trial_samples', level, wname, npc);
            npc(1:3) = zeros(1,3);
            [x_sim, qual, npc] = wmspca(trial_samples', level, wname, npc);
            x = x_sim';
            
			for j=1:channels
				row = x(j,:); % strip_artifacts(trial_samples(j,:), fs);
				features(1, j, i) = mean(row.^2);  % mean power of signal
			end
		end
		features = log(features); % logarithm of mean values
    end

	% build testing/training set
    train_data = compute(train_trials);
    test_data = compute(test_trials);
    
	% nothing changed - just copy labels
    train_lbl = train_labels;
    test_lbl = test_labels;
end


