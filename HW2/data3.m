function [ X, Y, mu, sig ] = data3(N)
    m1 = [1 1];
    m2 = [8 6];
    m3 = [13 1];
    sigma = 6 * eye(2);

    c1 = mvnrnd( m1, sigma, N(1));
    c2 = mvnrnd( m2, sigma, N(2));
    c3 = mvnrnd( m3, sigma, N(3));

    Y = [ones(N(1), 1); 2*ones(N(2), 1); 3*ones(N(3), 1)];
    X = [c1; c2; c3];
    mu = [mean(c1);mean(c2);mean(c3)];
    sig = {sigma; sigma; sigma};

end

