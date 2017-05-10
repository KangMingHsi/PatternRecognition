% 1 ========================================
% clear;

%generate data
g_mu = [1.0 3.0 2.0];
g_sigma = [0.1 0.1 0.2];
eps = 0.01;
data = [];
iteration = 100;

for i = 1:125
    data = [data double(normrnd(g_mu(2),g_sigma(2))) double(normrnd(g_mu(2),g_sigma(2))) double(normrnd(g_mu(1),g_sigma(1))) double(normrnd(g_mu(3),g_sigma(3)))];
end

mu = randi([0 3], [1 3]);
sigma = rand(1,3);
p = [1/3 1/3 1/3];

% % start iteration
 
old_ll = -Inf;
for t = 1:iteration
    c1 = [];
    c2 = [];
    c3 = [];

    for i = 1:length(data)

        Px1 = double(mvnpdf(data(i), mu(1), sigma(1)));
        Px2 = double(mvnpdf(data(i), mu(2), sigma(2)));
        Px3 = double(mvnpdf(data(i), mu(3), sigma(3)));

        div = Px1 * p(1) + Px2 * p(2) + Px3 * p(3);

        c1 = [c1 Px1*p(1)/div];
        c2 = [c2 Px2*p(2)/div];
        c3 = [c3 Px3*p(3)/div];
     end

     p = [mean(c1) mean(c2) 1-mean(c1)-mean(c2)];
     mu = [sum(c1 .* data)/sum(c1), sum(c2 .* data)/sum(c2), sum(c3 .* data)/sum(c3)];
     sigma = [sum(c1 .*((data-mu(1)).^2))/sum(c1), sum(c2 .* ((data-mu(2)).^2))/sum(c2), sum(c3 .* ((data-mu(3)).^2))/sum(c3)];
     
     ll = 0;
     for i = 1:length(data)

        Px1 = mvnpdf(data(i), mu(1), sigma(1));
        Px2 = mvnpdf(data(i), mu(2), sigma(2));
        Px3 = mvnpdf(data(i), mu(3), sigma(3));

        div = Px1 * p(1) + Px2 * p(2) + Px3 * p(3);

        ll = ll+log(div); 
     end

     if abs(ll - old_ll) <= eps
        break; 
     end
 
     old_ll = ll;
    
end
pause;
%% 2 ========================================
clear all;

N1 = 32; N2 = 256; N3 = 5000;
h1 = 0.05; h2 = 0.2;

data1 = linspace(0.0001,1.9999,N1);
data2 = linspace(0.0001,1.9999,N2);
data3 = linspace(0.0001,1.9999,N3);

x = -3:0.1:3;
% a
pa1 = zeros(1, length(x));
pa2 = zeros(1, length(x));
pa3 = zeros(1, length(x));

pb1 = zeros(1, length(x));
pb2 = zeros(1, length(x));
pb3 = zeros(1, length(x));


for i = 1:length(x)
    pa1(i) = sum(exp(-(x(i)-data1).^2 /(2*h1*h1)));
end
pa1 = pa1 / (N1 * sqrt(2 * pi) * h1);
figure; plot(x, pa1);


for i = 1:length(x)
    pa2(i) = sum(exp(-(x(i)-data2).^2 /(2*h1*h1)));
end
pa2 = pa2 / (N2 * sqrt(2 * pi) * h1);
figure; plot(x, pa2);

for i = 1:length(x)
    pa3(i) = sum(exp(-(x(i)-data3).^2 /(2*h1*h1)));
end
pa3 = pa3 / (N3 * sqrt(2 * pi) * h1);
figure; plot(x, pa3);

% b
for i = 1:length(x)
    pb1(i) = sum(exp(-(x(i)-data1).^2 /(2*h2*h2)));
end
pb1 = pb1 / (N1 * sqrt(2 * pi) * h2);
figure; plot(x, pb1);

for i = 1:length(x)
    pb2(i) = sum(exp(-(x(i)-data2).^2 /(2*h2*h2)));
end
pb2 = pb2 / (N2 * sqrt(2 * pi) * h2);
figure; plot(x, pb2);

for i = 1:length(x)
    pb3(i) = sum(exp(-(x(i)-data3).^2 /(2*h2*h2)));
end
pb3 = pb3 / (N3 * sqrt(2 * pi) * h2);
figure; plot(x, pb3);

pause;
%% 3 ========================================
clear all;

k1 = 32; k2 = 64; k3 = 256;
N = 5000;

data = linspace(0.0001,1.9999,N);
x = -3:0.1:3;

p1 = zeros(1,length(x));
p2 = zeros(1,length(x));
p3 = zeros(1,length(x));

for i = 1:length(x)
    tmp = sort(abs(x(i) - data));
    V = tmp(k1) * 2;
    p1(i) = k1 / N / V;
end
figure; plot(x, p1);

for i = 1:length(x)
    tmp = sort(abs(x(i) - data));
    V = tmp(k2) * 2;
    p2(i) = k2 / N / V;
end
figure; plot(x, p2);

for i = 1:length(x)
    tmp = sort(abs(x(i) - data));
    V = tmp(k3) * 2;
    p3(i) = k3 / N / V;
end
figure; plot(x, p3);

pause;
%% 4 ========================================
clear all;
data1 = mvnrnd([-5 0], eye(2), 100);
data2 = mvnrnd([5 0], eye(2), 100);

data = zeros(200, 3);
data(1:100, :) = [data1 zeros(100,1)+1];
data(101:200, :) = [data2 zeros(100,1)+1];

t = [zeros(100,1)+1; zeros(100,1)-1];
w = [0.1 0.1 0.1];

alpha = 0.5;
eps = 0.001;
round = 0;

% perceptron
while true
    round = round +1;
    Wrong = [];
    loss = 0;
    for i = 1:200
        tmp = w * data(i,:)';
        if(t(i) * tmp > 0)
            Wrong = [Wrong i];
            loss = loss + tmp * t(i) * -1;
        end
    end
    
    if(round > 10000 || loss < eps)
        break;
    end
    
    for i=1:length(Wrong)
        w = w + alpha * data(Wrong(i),:) * t(Wrong(i));
    end
end

x = -8:0.1:8;
y = -(w(1) * x + w(3))/w(2);

figure; plot(data1(:,1), data1(:,2), '+', data2(:,1), data2(:,2), '*', x,y, '-')
ylim([-8, 8]);

% Sum of square error
w = [0.1 0.1 0.1]';

X_plus = inv(data' * data) * data';
w = X_plus * t;
    
loss = sum((t - data * w).^2);

x = -8:0.1:8;
y = -(w(1) * x + w(3))/w(2);

figure; plot(data1(:,1), data1(:,2), '+', data2(:,1), data2(:,2), '*', x,y, '-')
ylim([-8, 8]);

% LMS
w = [0.1 0.1 0.1]';
alpha = 0.01;
eps = 0.001;
round = 0;

while true
    round = round +1;
    
    for i = 1:200
        w = w + (t(i) - data(i,:) * w) * alpha * data(i,:)';
    end
    
    if(round > 1000)
        break;
    end
    
end
x = -8:0.1:8;
y = -(w(1) * x + w(3))/w(2);

figure; plot(data1(:,1), data1(:,2), '+', data2(:,1), data2(:,2), '*', x,y, '-')
ylim([-8, 8]);
