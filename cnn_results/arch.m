
clc
clear all
close all
format compact

% processing times of each architecture (in sec) [from cnn_arch.log]
times = [
    351.0216 
    273.4448 
    558.0466 
    319.7162 
    568.1297 
    1142.0681
    459.9785
    1481.0904
    649.3152 
    635.8439 
    398.6974 
    1021.8841
] / 9; % average time per subject

S = load('cnn_arch.mat');
r = S.results;

figure
hold on
fprintf('arch,train,test\n');

d = numel(r);
train = zeros(d);
test = zeros(d);

for i=1:numel(r)
    o = r{i}; % o.x - 2x9
    y = mean(o.x,2);
    fprintf('%.2d,%.2f,%.2f\n', o.arch, y);
    train(i) = y(1);
    test(i) = y(2);
end

idx =setdiff(1:numel(r),7);
x = 1:numel(idx);

yyaxis left;
hold on;
plot(x,train(idx),'o-');
plot(x,test(idx),'v-');
ylabel('Accuracy');
axis([1 numel(x) 0.4 1]);

yyaxis right;
plot(x,times(idx),'s-');
ylabel('Processing time, s');
xlabel('Architecture');
legend('Training', 'Testing', 'Time');

