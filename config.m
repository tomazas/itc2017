function c = config(logfile)
    % default experiment configuration
    global LOG;
    
    if nargin > 0
        LOG = log4m.getLogger(logfile); % dump log to specific file
    else
        LOG = log4m.getLogger(); % init logger
    end
    
    addpath('utils/');
	addpath('2dmaps/');
    
	run('vlfeat/toolbox/vl_setup'); % initialize VLFeat library

	% default configuration
    c = {};
	c.crossval_func = @cnn_crossval;
	c.preprocess_func = @preprocess_bandpass;
	c.usable_channels = 1:22;
    c.datadir = 'E:/data/A0%dT.gdf.mat'; % 2a dataset EEG data
    c.cache_dir = 'data'; % directory for storing computational cache data
    c.n = 1:9;  % test subjects list
    c.fs = 250; % EEG signal sampling freq in Hz
    c.csp = true; % enable/disable CSP
    c.downsampling = 1; % >1 reduces/downsamples the EEG signal by specified amount
    c.skip_10fold = false; % enable/disable 10-fold validation
	c.skip_cache = false; % enable loading data from cache or not
	
    % usable EEG trial data range (in seconds) where the ERD/ERS/moto-imagery is happening
    c.trim_low = 2.5;
    c.trim_high = 5.5;
	
    % CNN default options
	%c.cnn_preview
	%c.preview_format
	c.cnn_preview_size = 1;
    c.cnn_img_size = [22 22]; % ractangular image size (in pixels)
	c.cnn_mapfunc = @map_replicated;
end