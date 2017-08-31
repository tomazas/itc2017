
clc
clear all
close all
format compact

S = load('cnn_filters.mat');
r = S.results;

figure
hold on
fprintf('size,train,test\n');

x = zeros(numel(r),1);
train = zeros(size(x));
test = zeros(size(x));

for i=1:numel(r)
    o = r{i}; % o.x - 2x9
    x(i) = o.filter(1);%sprintf('%dx%d',o.filter);
    y = mean(o.x,2);
    fprintf('%d,%.2f,%.2f\n', x(i), y);
    train(i) = y(1);
    test(i) = y(2);
end
plot(x,train,'o-');
plot(x,test,'v-');
xlabel('Filter size');
ylabel('Accuracy');
legend('Training', 'Testing');
axis tight;