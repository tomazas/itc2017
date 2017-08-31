
clc
clear all
close all
format compact

S = load('cnn_imgesize.mat');
r = S.results;

figure
hold on
fprintf('size,train,test\n');

x = zeros(numel(r),1);
train = zeros(size(x));
test = zeros(size(x));

for i=1:numel(r)
    o = r{i}; % o.x - 2x9
    x(i) = o.imgsize;
    y = mean(o.x,2);
    fprintf('%.2d,%.2f,%.2f\n', o.imgsize, y);
    train(i) = y(1);
    test(i) = y(2);
end
plot(x,train,'o-');
plot(x,test,'v-');
xlabel('Feature map size');
ylabel('Accuracy');
legend('Training', 'Testing');
axis tight;