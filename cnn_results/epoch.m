
clc
clear all
close all
format compact

S = load('cnn_epoch.mat');
r = S.results;

figure
hold on
fprintf('epoch,train,test\n');

x = zeros(numel(r),1);
train = zeros(size(x));
test = zeros(size(x));

for i=1:numel(r)
    o = r{i}; % o.x - 2x9
    x(i) = o.epoch;
    y = mean(o.x,2);
    fprintf('%.2f,%.2f,%.2f\n', o.epoch, y);
    train(i) = y(1);
    test(i) = y(2);
end
plot(x,train,'o');
plot(x,test,'v');
xlabel('Epoch');
ylabel('Accuracy');
legend('Training', 'Testing');