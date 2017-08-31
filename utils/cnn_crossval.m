function [err] = cnn_crossval(fun, data, labels, type, cv)
%     
% K-fold cross validation partition
%    NumObservations: 288
%        NumTestSets: 10
%          TrainSize: 260  259  259  259  259  259  259  259  259  260
%           TestSize: 28  29  29  29  29  29  29  29  29  28
%
      err = zeros(1, cv.NumTestSets);
      
      for i = 1:cv.NumTestSets
		  fprintf('  Processing x-fold: %d/%d\n',i,cv.NumTestSets);
          trIdx = cv.training(i);
          teIdx = cv.test(i);
          err(i) = fun(data(:,:,trIdx), labels(trIdx), data(:,:,teIdx), labels(teIdx));
      end
end