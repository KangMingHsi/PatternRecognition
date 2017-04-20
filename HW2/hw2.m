% 1. ===========================================
clear;

P = [1/3, 1/3, 1/3];
N = round(P * 1000);

[X Y mu,sigma] = data1(N);

Mdl = fitcnb(X, Y, 'ClassNames',{'1','2','3'});
drawPlot(X, Y, 1, mu, sigma, P);
L1 = loss(Mdl, X, Y, 'lossfun', 'classiferror')

L2 = classifier(X, Y,'euclidean', mu)
drawPlot(X, Y, 2, mu, sigma);

L3 = classifier(X, Y, 'mahalanobis', mu)
drawPlot(X, Y, 3, mu, sigma);

pause;
% 2. ===========================================
clear;

P = [1/3, 1/3, 1/3];
N = round(P * 1000);
[X Y mu sigma] = data2(N);

Mdl = fitcnb(X, Y, 'ClassNames',{'1','2','3'});
drawPlot(X, Y, 1, mu, sigma, P);
L1 = loss(Mdl, X, Y, 'lossfun', 'classiferror')

L2 = classifier(X, Y,'euclidean', mu)
drawPlot(X, Y, 2, mu, sigma);

L3 = classifier(X, Y, 'mahalanobis', mu)
drawPlot(X, Y, 3, mu, sigma);

pause;
% 3. ===========================================
clear;

P = [1/3, 1/3, 1/3];
N = round(P * 1000);
[X Y mu sigma] = data3(N);

Mdl = fitcnb(X, Y, 'ClassNames',{'1','2','3'});
drawPlot(X, Y, 1, mu, sigma, P);
L1 = loss(Mdl, X, Y, 'lossfun', 'classiferror')

L2 = classifier(X, Y,'euclidean', mu)
drawPlot(X, Y, 2, mu, sigma);

L3 = classifier(X, Y, 'mahalanobis', mu)
drawPlot(X, Y, 3, mu, sigma);

pause;
% 4. ===========================================
clear;

P = [1/3, 1/3, 1/3];
N = round(P * 1000);
[X Y mu sigma] = data4(N);

Mdl = fitcnb(X, Y, 'ClassNames',{'1','2','3'});
drawPlot(X, Y, 1, mu, sigma, P);
L1 = loss(Mdl, X, Y, 'lossfun', 'classiferror')

L2 = classifier(X, Y,'euclidean', mu)
drawPlot(X, Y, 2, mu, sigma);

L3 = classifier(X, Y, 'mahalanobis', mu)
drawPlot(X, Y, 3, mu, sigma);

pause;
% 5. ===========================================
clear;



pause;
% 6. ===========================================
clear

P1 = [1/3 1/3 1/3];
P2 = [0.8 0.1 0.1];
N1 = round(P1 * 1000);
N2 = round(P2 * 1000);

[X1 Y1 mu1 sigma1] = data5(N1);
[X2 Y2 mu2 sigma2] = data5(N2);

Mdl1 = fitcnb(X1, Y1, 'ClassNames',{'1','2','3'});
drawPlot(X1, Y1, 1, mu1, sigma1, P1);
L11 = loss(Mdl1, X1, Y1, 'lossfun', 'classiferror')

Mdl2 = fitcnb(X2, Y2, 'ClassNames',{'1','2','3'});
drawPlot(X2, Y2, 1, mu2, sigma2, P2);
L12 = loss(Mdl2, X2, Y2, 'lossfun', 'classiferror')

L21 = classifier(X1, Y1,'euclidean', mu1)
drawPlot(X1, Y1, 2, mu1, sigma1);

L22 = classifier(X2, Y2,'euclidean', mu2)
drawPlot(X2, Y2, 2, mu2, sigma2);
