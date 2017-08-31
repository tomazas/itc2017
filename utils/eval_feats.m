% extract features, classify them and verify correctness using 10xfold crossvalidation
function [training_err, testing_err, tenfold_train_err, tenfold_test_err] = eval_feats(p, feat_func, class_func, train_trials, test_trials, train_labels, test_labels)
    global LOG
    
    if ~exist(p.cache_file, 'file') || p.skip_cache
        % single training+classification
        [train_data, test_data, train_lbl, test_lbl] = feat_func(p, train_trials, test_trials, train_labels, test_labels);
        
        LOG.info('>>> Saving to cache: %s', p.cache_file);
        save(p.cache_file, 'train_data', 'test_data', 'train_lbl', 'test_lbl');
    else
        % load cached features - speeds up computations
        LOG.info('>>> Using cache: %s', p.cache_file);
        load(p.cache_file);
    end
    
    [c, training_err] = class_func(p, test_data, train_data, train_lbl, test_lbl);
    
    if isfield(p, 'cnn_preview')
        training_err = 0;
        testing_err = 0;
        tenfold_train_err = zeros(1,10);
        tenfold_test_err = zeros(1,10);
        return; % quit early
    end
    
    % compute testing error
    bad = sum(bsxfun(@ne, c, test_lbl)); %  compare classes with true labels
    testing_err = sum(bad) / length(test_lbl); % error ratio
    
    lbl_unique = unique(test_lbl);
    c_unique = unique(c);
    assert(isempty(setdiff(c_unique,lbl_unique)), 'Predicted classes and label classes must match!');
    
    C = confusionmat(single(test_lbl), single(c))
    
    % do 10-fold cross validation
    if ~isfield(p, 'skip_10fold') || ~p.skip_10fold
    	rng(0,'twister');
        fprintf('Starting 10-fold validation for T%d\n', p.subjectNo);
        
        cp = cvpartition(train_lbl, 'k', 10); % generate 10 disjoint stratified subsets
        fun = @(xTrain,yTrain,xTest,yTest)(sum(class_func(p, xTest, xTrain, yTrain, yTest) ~= yTest));
        tenfold_train_err = p.crossval_func(fun, train_data, train_lbl, 'partition', cp);
        tenfold_train_err = tenfold_train_err ./ cp.TestSize

        cp = cvpartition(test_lbl, 'k', 10); % generate 10 disjoint stratified subsets
        fun = @(xTrain,yTrain,xTest,yTest)(sum(class_func(p, xTest, xTrain, yTrain, yTest) ~= yTest));
        tenfold_test_err = p.crossval_func(fun, test_data, test_lbl, 'partition', cp);
        tenfold_test_err = tenfold_test_err ./ cp.TestSize
    else
        tenfold_train_err = zeros(1,10);
        tenfold_test_err = zeros(1,10);
        fprintf('Skipping 10-fold validation (disabled)\n');
    end
end
