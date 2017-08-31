function validate_cnn(p, train_data, train_labels, test_data)
	assert(size(train_data,1) > size(train_data,2), 'Train data must be row based!');
	assert(size(train_labels,1) > size(train_labels,2), 'Labels must be stored in rows!');
	
	cd('../'); % this script is called from a subfolder, so required to goto root directory
	addpath('features/');
    addpath('classifiers/')
	addpath('utils/');
	
	% shuffle the data
	shuffle_idx = randperm(length(train_labels));
	train_labels = train_labels(shuffle_idx);
	train_data = train_data(shuffle_idx,:);

	[test_classes, training_err] = classify_conv(p, test_data, train_data, train_labels);
	fprintf('training accuracy = %.2f\n', (1-training_err)*100);
end