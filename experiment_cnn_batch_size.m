clc
clear all
close all
format compact

global LOG

p = config('cnn_batch.log');
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
outname = 'cnn_batch';
batches = [10 20 30 50 75 100 150 175 200 250 300];
results = cell(1, numel(batches));

elapsed = [];

t_init = tic();
for i=1:numel(batches)
    t0 = tic();
    
    p.mat_cnn_options = trainingOptions(...
          'sgdm', ...
          'InitialLearnRate', 0.01, ...
          'Momentum', 0.1, ...
          'Verbose', true, ...
          'Shuffle', 'once', ...
          'MiniBatchSize', batches(i), ...
          'MaxEpochs', 500);
    
    LOG.info('Begin batch = %d evaluation', batches(i)); 
    [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

    r = {};
    r.batch = batches(i);
    r.feats = @mean_signal_power;
    r.class = @classify_matcnn;
    r.x = x;
    r.xfold = xfold;
    
    results{i} = r;
    save(sprintf('cnn_results/%s.mat', outname), 'results');
    
    elapsed = mean([elapsed toc(t0)]);
    LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
    i, numel(batches), elapsed, (numel(batches)-i) * elapsed);
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
