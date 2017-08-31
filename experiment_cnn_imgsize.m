clc
clear all
close all
format compact

global LOG

p = config('cnn_imgsize.log');
p.skip_10fold = true;

p.mat_cnn_options = trainingOptions(...
      'sgdm', ...
      'InitialLearnRate', 0.01, ...
      'Momentum', 0.01, ...
      'Verbose', true, ...
      'Shuffle', 'once', ...
      'MiniBatchSize', 128, ...
      'MaxEpochs', 200);
        
%% run experiment
outname = 'cnn_imgsize';
imgsz = [8 10 12 16 20 22 24 28 30 32 36 40 44 49 50 64];
results = cell(1,numel(imgsz));
elapsed = [];

t_init = tic();
for i=1:numel(imgsz)
    t0 = tic();
    
    p.cnn_img_size = [imgsz(i) imgsz(i)];
    
    % matlab CNN
    p.mat_cnn_layers = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
    
    LOG.info('Begin imgsize = [%d %d] evaluation', p.cnn_img_size); 
    [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

    r = {};
    r.imgsize = imgsz(i);
    r.feats = @mean_signal_power;
    r.class = @classify_matcnn;
    r.x = x;
    r.xfold = xfold;
    
    results{i} = r;
    save(sprintf('cnn_results/%s.mat', outname), 'results');
    
    elapsed = mean([elapsed toc(t0)]);
    LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
    i, numel(imgsz), elapsed, (numel(imgsz)-i) * elapsed);
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
