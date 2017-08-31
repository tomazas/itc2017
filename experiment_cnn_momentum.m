clc
clear all
close all
format compact

global LOG

p = config('cnn_momentum.log');
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
outname = 'cnn_momentum';
momentums = 0.01:0.1:1;

results = cell(1, numel(momentums));
elapsed = [];

t_init = tic();
for i=1:numel(momentums)
    t0 = tic();
    
    p.mat_cnn_options = trainingOptions(...
          'sgdm', ...
          'InitialLearnRate', 0.01, ...
          'Momentum', momentums(i), ...
          'Verbose', true, ...
          'Shuffle', 'once', ...
          'MiniBatchSize', 128, ...
          'MaxEpochs', 500);
    
    LOG.info('Begin momentum = %f evaluation', momentums(i)); 
    [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

    r = {};
    r.momentum = momentums(i);
    r.feats = @mean_signal_power;
    r.class = @classify_matcnn;
    r.x = x;
    r.xfold = xfold;
    
    results{i} = r;
    save(sprintf('cnn_results/%s.mat', outname), 'results');
    
    elapsed = mean([elapsed toc(t0)]);
    LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
    i, numel(momentums), elapsed, (numel(momentums)-i) * elapsed);
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
