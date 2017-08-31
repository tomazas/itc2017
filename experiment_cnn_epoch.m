clc
clear all
close all
format compact

global LOG

p = config('cnn_epoch.log');
p.skip_10fold = true;

% matlab CNN
p.mat_cnn_layers = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([4 4],16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

%% run experiment
outname = 'cnn_epoch';
epochs = [50 100 200 300 500 700 1000 1200 1500 2000 2500];
results = cell(1,numel(epochs));
elapsed = [];

t_init = tic();
for i=1:numel(epochs)
    t0 = tic();
    
    p.cnn_numEpochs = epochs(i);
    p.mat_cnn_options = trainingOptions(...
          'sgdm', ...
          'InitialLearnRate', 0.01, ...
          'Momentum', 0.1, ...
          'Verbose', true, ...
          'Shuffle', 'once', ...
          'MiniBatchSize', 128, ...
          'MaxEpochs', p.cnn_numEpochs);
    
    LOG.info('Begin epoch = %d evaluation', p.cnn_numEpochs); 
    [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

    r = {};
    r.epoch = p.cnn_numEpochs;
    r.feats = @mean_signal_power;
    r.class = @classify_matcnn;
    r.x = x;
    r.xfold = xfold;
    
    results{i} = r;
    save(sprintf('cnn_results/%s.mat', outname), 'results');
    
    elapsed = mean([elapsed toc(t0)]);
    LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
    i, numel(epochs), elapsed, (numel(epochs)-i) * elapsed);
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
