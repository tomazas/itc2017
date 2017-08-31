function v = multiclass_csp(s, h, params)
    pre = params.trim_low;
    post = params.trim_high;

    fs = h.SampleRate;

    fprintf('Multiclass CSP calculated for signal, pre = %d, post = %d\n', pre, post);
    t0 = round(pre*fs);
    t1 = round(post*fs);
    
    % extract samples for each class
    c1 = trigg(s, h.TRIG(h.Classlabel==1), t0, t1)';
    c2 = trigg(s, h.TRIG(h.Classlabel==2), t0, t1)';
    c3 = trigg(s, h.TRIG(h.Classlabel==3), t0, t1)';
    c4 = trigg(s, h.TRIG(h.Classlabel==4), t0, t1)';

    % build covariance matrix
    cov1 = covm(c1,'E');
    cov2 = covm(c2,'E');
    cov3 = covm(c3,'E');
    cov4 = covm(c4,'E');
    n = size(cov1,1);
    
    ECM = zeros(4,n,n);
    ECM(1,:,:) = cov1;
    ECM(2,:,:) = cov2;
    ECM(3,:,:) = cov3;
    ECM(4,:,:) = cov4;
    
    %ECM = cat(3,covm(c1,'E'),covm(c2,'E'), covm(c3,'E'), covm(c4,'E'));
    %ECM = permute(ECM,[3,1,2]);

    v = csp3(ECM); % multiclass CSP using One-vs-Rest scheme
end
