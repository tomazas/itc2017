clc
clear all
close all
format compact

global LOG

p = config('cnn_arch.log');
p.skip_10fold = true;

p.mat_cnn_options = trainingOptions(...
      'sgdm', ...
      'InitialLearnRate', 0.01, ...
      'Momentum', 0.01, ...
      'Verbose', true, ...
      'Shuffle', 'once', ...
      'MiniBatchSize', 128, ...
      'MaxEpochs', 500);
        
%% run experiment
outname = 'cnn_arch';
p.cnn_img_size = [44 44];

% architecture definitions
a1 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 4);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
a2 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 4);
            reluLayer();
            maxPooling2dLayer(2,'Stride',4);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
a3 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 8);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
a4 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

a5 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 32);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
a6 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 64);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

% single/double size architectures
b1 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
b2 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            convolution2dLayer([7 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

% no max pooling layer      
c1 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 16);
            reluLayer();
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
% no ReLU layer   
d1 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 7], 16);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

% row filters
e1 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([7 1], 16);
            reluLayer();
            convolution2dLayer([1 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];

e2 = [imageInputLayer([p.cnn_img_size 1]);
            convolution2dLayer([1 7], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            convolution2dLayer([7 1], 16);
            reluLayer();
            maxPooling2dLayer(2,'Stride',2);
            fullyConnectedLayer(4);
            softmaxLayer();
            classificationLayer()];
        
archi = {a1 a2 a3 a4 a5 a6 b1 b2 c1 d1 e1 e2};
results = cell(1,numel(archi));
elapsed = [];

t_init = tic();
for i=1:numel(archi)   
    try
        % matlab CNN
        t0 = tic();
        p.mat_cnn_layers = archi{i};

        LOG.info('Begin arch = %d evaluation', i); 
        [x, xfold] = run_experiment(@mean_signal_power, @classify_matcnn, p);

        r = {};
        r.arch = i;
        r.feats = @mean_signal_power;
        r.class = @classify_matcnn;
        r.x = x;
        r.xfold = xfold;

        results{i} = r;
        save(sprintf('cnn_results/%s.mat', outname), 'results');

        elapsed = mean([elapsed toc(t0)]);
        LOG.info('Evaluation step %d/%d done. Elapsed: %.4f sec, approx. time remaining: %.4f sec', ...
            i, numel(archi), elapsed, (numel(archi)-i) * elapsed);
     catch e
         LOG.error('failure: %s', e.message);
         display(e);
     end
end

LOG.info('Done. Total time: %.4f sec', toc(t_init));
