clc
clear all
close all
format compact

global LOG

p = config('cnn_learn_rate.log');
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
outname = 'cnn_learn_rate';
rate = [0.001 0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.2 0.5];

results = cell(1, numel(rate));
elapsed = [];

t_init = tic();
for i=1:numel(rate)
    t0 = tic();
    
    p.mat_cnn_options = trainingOptions(...
          'sgdm', ...
          'InitialLearnRate', rate(i), ...
          'Momentum', 0.01, ...
          'Verbose', true, ...
          'Shuffle', 'once', ...
          'MiniBatchSize', 128, ...
          'MaxEpochs', 500);
    
    LOG.info('Begin learn rate = %f evaluation', rate(i)); 
    [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

    r = {};
    r.rate = rate(i);
    r.feats = @mean_signal_power;
    r.class = @classify_matcnn;
    r.x = x;
    r.xfold = xfold;
    
    results{i} = r;
    save(sprintf('cnn_results/%s.mat', outname), 'results');
    
    elapsed = mean([elapsed toc(t0)]);
    LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
    i, numel(rate), elapsed, (numel(rate)-i) * elapsed);
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
