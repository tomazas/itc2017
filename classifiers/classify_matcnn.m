function [test_classes, training_err] = classify_matcnn(p, test_data, train_data, train_labels, test_labels)

    % prepares CNN images based on selected algorithm
    function data = prepareImages(sample_data, img_sz)
        data = p.cnn_mapfunc(sample_data, img_sz);
        fprintf('Images prepared: %d of size: %d x %d\n', size(data,4), size(data,1), size(data,2));
    end

	% CNN network testing (evaluation) function
    function [err, classes] = evaluate_net(network, images, labels)
        classes = classify(network, images);
        err = sum(labels ~= classes)/numel(classes);
        classes = single(classes);
    end

    % extra preprocessing step
    function [x, Q] = prepare_features(data, labels, Q)
        if isempty(labels)
            x = preprocess(data, [], Q);
        else
            [x, Q] = preprocess(data, labels);
        end
    end

    function out = prepare_labels(inp)
        out = categorical(inp);
    end

    % --------------------------------------------------------------------
    % Setup training
    % --------------------------------------------------------------------
    
    % data to be classified
    num_classes = length(unique(train_labels));
    assert(num_classes == 4, 'Configured to solve only 4-class problems, check your labels!');
    
    % print info
    [s,h,d] = size(train_data); % ch x samples x trials
    fprintf('CNN classes: %d, train data: %d x %d x %d, labels: %d\n', num_classes, s, h, d, length(train_labels));
    
    [s,h,d] = size(test_data);
	fprintf('CNN test data: %d x %d x %d\n', s, h, d);
	
    % prepare training data
    [pre_train_data, Q] = prepare_features(train_data, train_labels);
    prepared_train_data = prepareImages(pre_train_data, p.cnn_img_size);
    prepared_train_labels = prepare_labels(train_labels);
    
    % we cannot classify if the data is wrong
    assert(length(prepared_train_labels) == size(prepared_train_data, 4), 'Number of labels and number of train images must match!');
    
    % prepare for testing
    [pre_test_data, ~] = prepare_features(test_data, [], Q);
    prepared_test_data = prepareImages(pre_test_data, p.cnn_img_size);
    prepared_test_labels = prepare_labels(test_labels);
    
    % we cannot classify if the data is wrong
    assert(length(prepared_test_labels) == size(prepared_test_data, 4), 'Number of labels and number of test images must match!');
    
    % preview (save the generated images to pictures in the filesystem
    if isfield(p, 'cnn_preview')
        if ~exist('./cnn_img', 'dir'), mkdir('./cnn_img'); end
        colormap(jet(64));
        
		num_img = min(max(1,p.cnn_preview_size), size(prepared_train_data,4));
        for i=1:num_img
            fprintf('output image %d/%d\n', i, num_img);
            im = prepared_train_data(:,:,:,i);
            imagesc(im);
            if isfield(p, 'preview_format')
                eval(p.preview_format); % use custom formatting commands
            end
            saveas(gcf, sprintf(p.cnn_preview, p.subjectNo, i));
        end
        
        % exit quickly
        training_err = 0;
        test_classes = zeros(size(test_labels));
        return;
    end
    
    rng('default'); % For reproducibility 

    % --------------------------------------------------------------------
    % Do training
    % --------------------------------------------------------------------
    [net, traininfo] = trainNetwork(prepared_train_data, prepared_train_labels, p.mat_cnn_layers, p.mat_cnn_options);

    % classify all training images & compute training error
    [training_err, ~] = evaluate_net(net, prepared_train_data, prepared_train_labels);
    fprintf('CNN training error: %.f%%\n', training_err*100);

	% classify all test images
    [testing_err, test_classes] = evaluate_net(net, prepared_test_data, prepared_test_labels);
    fprintf('CNN testing error: %.f%%\n', testing_err*100);
end


