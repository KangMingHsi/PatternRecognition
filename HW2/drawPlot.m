function [] = drawPlot( X, Y, mode, mu, sigma, P)

    // get the drawing range
    xrange = [min(X(:,1)) max(X(:,1))];
    yrange = [min(X(:,2)) max(X(:,2))];
    inc = 0.1;

    [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
    image_size = size(x);
    
    xy = [x(:) y(:)];
    numxypairs = length(xy);
    
    sample_means = cell(size(mu,1), 1);

    for i = 1:length(mu)
        sample_means{i} = mu(i,:);
    end
    
    figure;
    hold on

    // for different classifier, use different method
    // try to iterate all the 2D-point that we are going to plot
    // use the classifier to know which class it should be
    // and then draw the color on the point
    if mode == 1
        
        dist = [];
        for i=1:length(mu),
            t = xy - repmat(sample_means{i}, [numxypairs 1]);
            inverse = repmat(inv(sigma{i}), [numxypairs 1]);
            disttemp = [];
            for j = 1:length(t)
            
                temp = exp(-(t(j,:) * inverse(2*(j-1)+1:2*j, :) * t(j,:)') / 2) ...
                    / (2 * pi * sqrt(det(sigma{i})));    %sum((xy - repmat(sample_means{i}, [numxypairs 1])) .^ 2, 2);
                temp = temp * P(i);
                disttemp = [disttemp; temp];
            end
            
            dist = [dist disttemp]; 
        end
        
        [m,idx] = max(dist, [], 2);
        decisionmap = reshape(idx, image_size);
        
        axis([xrange(1) xrange(2) yrange(1) yrange(2)]);
        imagesc(xrange,yrange,decisionmap);
        set(gca,'ydir','normal');
        cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
        colormap(cmap);
        
        title('Naive Bayes Classifier');
    elseif mode == 2
        
        dist = [];
        for i=1:length(mu),
            disttemp = sum((xy - repmat(sample_means{i}, [numxypairs 1])) .^ 2, 2);
            dist = [dist disttemp]; 
        end
        
        [m,idx] = min(dist, [], 2);
        decisionmap = reshape(idx, image_size);
        
        axis([xrange(1) xrange(2) yrange(1) yrange(2)]);
        imagesc(xrange,yrange,decisionmap);
        set(gca,'ydir','normal');
        cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
        colormap(cmap);
        
        title('Euclidean Classifier');
    else
        dist = [];
        for i=1:length(mu),
 
            disttemp = sum(abs(xy - repmat(sample_means{i}, [numxypairs 1])), 2);

            dist = [dist disttemp]; 
        end
        [m,idx] = min(dist, [], 2);
        decisionmap = reshape(idx, image_size);
        
        axis([xrange(1) xrange(2) yrange(1) yrange(2)]);
        imagesc(xrange,yrange,decisionmap);
        set(gca,'ydir','normal');
        cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
        colormap(cmap);
        
        title('Mahalanobis Classifier');
    end
    
    gscatter(X(:,1),X(:,2),Y);
    h = gca;
    h.XLim = [xrange(1) xrange(2)];
    h.YLim = [yrange(1) yrange(2)];
    
    xlabel('X');
    ylabel('Y');
    hold off

end

