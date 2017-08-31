clc
clear all
close all
format compact

global LOG

p = config('cnn_bilinear.log');

p.cnn_img_size = [44 44];

% matlab CNN
p.mat_cnn_layers = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7],16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

p.mat_cnn_options = trainingOptions(...
          'sgdm', ...
          'InitialLearnRate', 0.01, ...
          'Momentum', 0.01, ...
          'Verbose', true, ...
          'Shuffle', 'once', ...
          'MiniBatchSize', 128, ...
          'MaxEpochs', 500);

%% run experiment
feats = {
    @plain_fft_features, ...
    @plain_fft_features_bilinear, ...
    @plain_fft_features_bicubic, ...
    @plain_signal_features, ...
    @plain_signal_features_bilinear, ...
    @plain_signal_features_bicubic
};

outname = 'xfold_bilinear';
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
