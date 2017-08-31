% plots raw EEG data signal and also a 2D image
function eegplot(eeg, ftitle)
    eeg = normalize(eeg,1);
	[ch, samples] = size(eeg);
	if ch > samples
		eeg = eeg';
		[ch, samples] = size(eeg);
    end
	
	figure;
	hold on;
	for i=1:ch
        plot(eeg(i, :) + i);
    end
	
    if nargin > 1
        title(sprintf('EEG signal plot (%s)', ftitle));
    else
        title('EEG signal plot');
    end
    xlabel('Samples');
    ylabel('Channels');
	hold off;
    axis tight;
    
    figure;
    imagesc(eeg);
    set(gca,'YDir','normal'); % the image is inverted - revert it!
    title('EEG signal image');
    xlabel('Samples');
    ylabel('Channels');
end