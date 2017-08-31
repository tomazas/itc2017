clc
clear all
close all
format compact

global LOG

p = config('cnn_xfold.log');

% matlab CNN
p.mat_cnn_layers = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([4 4],16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

p.mat_cnn_options = trainingOptions(...
          'sgdm', ...
          'InitialLearnRate', 0.01, ...
          'Momentum', 0.1, ...
          'Verbose', true, ...
          'Shuffle', 'once', ...
          'MiniBatchSize', 128, ...
          'MaxEpochs', 200);

%% run experiment
feats = {
    @mean_signal_power, ...	
    @mean_var_features, ...	
    @mean_window_power, ...	
    @pca_features, ...		
    @mean_bp_features,... 	
    @fft_power_features,... 	
    @dct_features,... 		
    @mean_tdp_features,... 	
    @mean_teager_features,... 
    @plain_fft_features,... 	
    @plain_wavelet_features,...
    @plain_signal_features, ...
    @energy_map_features, ...
};

outname = 'xfold_all';
results = cell(1,numel(feats));
elapsed = [];

t_init = tic();
for i=1:numel(feats)
    try
        t0 = tic();
        func = feats{i};
        
        LOG.info('Evaluating: %s', func2str(func));
        [x, xfold] = run_experiment(func, @classify_matcnn, p);
        
        r = {};
        r.feats = func;
        r.class = @classify_matcnn;
        r.x = x;
        r.xfold = xfold;
        results{i} = r;
        save(sprintf('cnn_results/%s.mat', outname), 'results');
        
        elapsed = mean([elapsed toc(t0)]);
        LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
            i, numel(feats), elapsed, (numel(feats)-i) * elapsed);
    catch e
        LOG.error('exception caught: %s', e.message);
        display(e);
    end
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
