% prepares images by copying the samples to form an (img_size x img_size) rectangles
% each sampleData trial corresponds to one image
% sampleData - input EEG data [ch x samples x trials]
% out - output 4D array [width x height x 1 x trials]
function [out] = map_replicated(sampleData, imgSize)
    fprintf('Preparing CNN replicated images\n');
    
    % samples will be duplicated to form same amount of images
    [ch, samples, trials] = size(sampleData);
    
	expanded = sampleData;
	
    % exapand data to side to get at least required image size - `img_sz`
	if samples < imgSize(1)
        repeat = ceil(imgSize(1)/samples);
		expanded = repmat(expanded, [1 repeat]);
	end
	
	expanded = expanded(:,1:imgSize(1),:); % limit data to form full image size rectangles (crop in width)
	
    if ch < imgSize(2)
        repeat = ceil(imgSize(2)/ch);
        out = zeros(repeat*ch, imgSize(1), trials);
        
        for i=1:trials
            out(:,:,i) = kron(expanded(:,:,i), ones(repeat,1)); % duplicate each sample row by `height` times
        end
        expanded = out(1:imgSize(2),:,:); % limit data in height
    else
        expanded = expanded(1:imgSize(2),:,:); % limit data in height
    end

    %form a 4D image array
    out = reshape(expanded, imgSize(2), imgSize(1), 1, trials); % ensure we have 4D array (every image with single layer)
end
