% Computes EEG power using wavelet convolution
% input - single EEG trial data
% fs - signal sampling frequency
% ntrials - number of trials
% nsamples - number of samples in single trial
% num_frex - number of frequencies to compute
% showplot - true, if plotting of result is required
%
% returns computed time-frequency power (tf) and phase (tp)
function [tf, tp] = compute_wavepower(input, fs, ntrials, nsamples, num_frex, showplot)
	% wavelet parameters
	freq =  [0 30];
	cycles = [0.5 5]; % wavelet cycles

	% other wavelet parameters
	frex  = linspace(freq(1), freq(end), num_frex);
	nCycs = linspace(cycles(1), cycles(end), num_frex);
	time  = -2:1/fs:2;
	half_wave = (length(time)-1)/2;

	% FFT parameters
	nWave = length(time);
	nData = nsamples*ntrials;
	nConv = nWave+nData-1;

	% FFT of data (doesn't change on frequency iteration)
	flatteneddata = reshape(input, 1, nData);
	dataX = fft(flatteneddata, nConv);

	% initialize output time-frequency data
	tf = zeros(num_frex, nsamples);
    tp = zeros(num_frex, nsamples);
    
	% loop over frequencies
    for fi=1:num_frex
		% create wavelet and get its FFT
		s = nCycs(fi)/(2*pi*frex(fi));
		wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2)); % convolve with gaussian
		waveletX = fft(wavelet,nConv); % fft of wavelet

		% run convolution in frequency domain
		as = ifft(waveletX.*dataX,nConv);
		as = as(half_wave+1:end-half_wave); % chop of half wavelet
		as = reshape(as, nsamples, ntrials);
        
		% compute ITPC
        tf(fi,:) = mean(abs(as),2); % this is for power
        tp(fi,:) = abs(mean(exp(1i*angle(as)), 2)); % this is for phase
    end
	
    if nargin > 5 && showplot
        % plot results
        figure, clf
        contourf(1:size(tf,2),frex,tf,40,'linecolor','none')
        %set(gca,'clim',[0 .6],'ydir','normal','xlim',[-300 1000])
        title('Energy map')
        xlabel('Time (ms)'), ylabel('Frequency (Hz)')
        
        figure, clf
        contourf(1:size(tp,2),frex,tp,40,'linecolor','none')
        %set(gca,'clim',[0 .6],'ydir','normal','xlim',[-300 1000])
        title('Phase map')
        xlabel('Time (ms)'), ylabel('Frequency (Hz)')
    end
end