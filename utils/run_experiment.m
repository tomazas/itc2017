% runs all tests for passed feature function and classifier function
function [results, xfold_results] = run_experiment(feat_func, class_func, p)

	addpath('features/');
    addpath('classifiers/')
	addpath('utils/');
    
	global LOG
    
    % dump config parameters
    p

	% structure of the signal
    % S = struct:
    %             sig1: [672528x22 double]
    %             sig2: [687000x22 double]
    %      train_feats: [22x626x288 double]
    %       test_feats: [22x626x281 double]
    %     train_labels: [288x1 double]
    %      test_labels: [281x1 double]
    %               h1: [1x1 struct]
    %               h2: [1x1 struct]
    
    results = zeros(2,numel(p.n));
    xfold_results = zeros(2,10,numel(p.n));
    
    t0 = tic();
    
    for i = p.n
        t1 = tic();
        
        % store extra info for logging/image generation
        p.subjectNo = i;
        p.feat_func = feat_func;
        
        p.cache_file = sprintf('%s/%s_T%d.mat', p.cache_dir, func2str(feat_func), i);
        
        if ~exist(p.cache_file, 'file') || p.skip_cache
            filename = sprintf(p.datadir, i);
            LOG.info('>>> Cache not found, loading signal: %s', filename);
            S = load(filename);

            LOG.info('  Extracting features...');
            p.csp_matrix = [];
            [~, train_trials, csp_mat] = get_trials(S.sig1, S.h1, p);        
            p.csp_matrix = csp_mat; % save computed CSP matrix for later use
            [~, test_trials] = get_trials(S.sig2, S.h2, p);
            
            LOG.info('  Removing artifacts from test data...');
            test_trials = test_trials(:,:,S.h2.ArtifactSelection==0); % TODO: preprocess also test_trials in data files and remove this
            %S.test_labels = S.test_labels(S.h2.ArtifactSelection==0); % S.test_labels are already preprocessed
            
            % normalize: 0 mean and unit variance
            train_trials = normalize(train_trials);
            test_trials = normalize(test_trials);
            
            train_labels = S.train_labels;
            test_labels = S.test_labels;
        else
            LOG.info('>>> Loading from cache scheduled');
            
            % data will be filled later
            train_trials = [];
            test_trials = [];
            train_labels = [];
            test_labels = [];
        end

		% allow to do some other computation on input data (e.g. analysis)
        if isfield(p, 'analyze')
            LOG.info('Analyzing data with: %s', func2str(p.analyze));
            p.analyze(train_trials, test_trials, S.train_labels, S.test_labels);
            continue;
        end

		LOG.info('Evaluating: features - %s, classifier - %s...', func2str(feat_func), func2str(class_func));
        [training_err, testing_err, tfold_train_err, tfold_test_err] = eval_feats(p, feat_func, class_func, ...
            train_trials, test_trials, train_labels, test_labels);
        
        if isfield(p, 'cnn_preview')
            return; % quit early
        end
        
        timesub = toc(t1);
        LOG.info('Subject time: %.4f sec, approx time ramaining: %.4f sec', timesub, (numel(p.n)-i) * timesub);
      
        LOG.info('Iteration %d:', i);
        train_accuracy = 1 - training_err;
        test_accuracy = 1 - testing_err;
        LOG.info('  Classify accuracy - train: %.4f test: %.4f', train_accuracy, test_accuracy);
        
        tfold_train_accuracy = ones(1, numel(tfold_train_err)) - tfold_train_err;
        tfold_test_accuracy = ones(1, numel(tfold_test_err)) - tfold_test_err;
        LOG.info('  Validate 10-fold  - train: %.4f test: %.4f\n', mean(tfold_train_accuracy), mean(tfold_test_accuracy));
        
        results(:,i) = [train_accuracy; test_accuracy]
        xfold_results(:,:,i) = [tfold_train_accuracy; tfold_test_accuracy]
    end
    
    elapsed = toc(t0);
    LOG.info('Processed in: %.4f sec', elapsed);
    
    LOG.info('mean: %.4f/%.4f\n', mean(results,2));
    
    if ~p.skip_10fold
        LOG.info('mean 10-fold: %.4f/%.4f\n', mean(mean(xfold_results,2),3));
    end
end



