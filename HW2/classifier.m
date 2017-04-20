function [ Error] = classifier(X, Y, mode, mu)
     // count how many wrong clssify
    ErrorCount = 0;
    if strcmp(mode,'euclidean')
        for i = 1:length(Y)
            
            record = zeros(length(mu),1);

            // use euclidean method
            for j = 1:length(mu)
               record(j) = sum((X(i,:) - mu(j,:)).^2,2);
            end

            // classify it to the shortest distance class
            [v idx] = min(record);

            // if the classify result is not the same as the label
            // error increase
            if idx ~= Y(i)
               
               ErrorCount = ErrorCount+1; 
               
            end
        end
    // as above only for the different method
    elseif strcmp(mode, 'mahalanobis')
        for i = 1:length(Y)
            
            record = zeros(length(mu),1);
            for j = 1:length(mu)
               record(j) = sum(abs(X(i,:) - mu(j,:)),2);
            end
            [v idx] = min(record);
            
            if idx ~= Y(i)
               ErrorCount = ErrorCount+1; 
            end
        end
    end

    // divided by all samples to calculate the error rate
    Error = double(ErrorCount) / length(Y);
end

