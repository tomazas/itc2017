clc
clear all
close all
format compact

global LOG

p = config('cnn_filters.log');
p.skip_10fold = true;

p.mat_cnn_options = trainingOptions(...
      'sgdm', ...
      'InitialLearnRate', 0.01, ...
      'Momentum', 0.1, ...
      'Verbose', true, ...
      'Shuffle', 'once', ...
      'MiniBatchSize', 128, ...
      'MaxEpochs', 200);
        
%% run experiment
outname = 'cnn_filters';
filters = {[2 2], [3 3], [4 4], [5 5], [6 6], [7 7], [8 8], [9 9], [10 10], [11 11]};
results = cell(1,numel(filters));
elapsed = [];

t_init = tic();
for i=1:numel(filters)
    t0 = tic();
    
    % matlab CNN
    p.mat_cnn_layers = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer(filters{i}, 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
    
    LOG.info('Begin filter = [%d %d] evaluation', filters{i}); 
    [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

    r = {};
    r.filter = filters{i};
    r.feats = @mean_signal_power;
    r.class = @classify_matcnn;
    r.x = x;
    r.xfold = xfold;
    
    results{i} = r;
    save(sprintf('cnn_results/%s.mat', outname), 'results');
    
    elapsed = mean([elapsed toc(t0)]);
    LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
    i, numel(filters), elapsed, (numel(filters)-i) * elapsed);
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
