function convert(filename,class)

%read the input file
fprintf('Reading file: %s\n',filename)
fid = fopen(filename);

textLine = fgets(fid); % Read first line
lineCounter = 1;

while ischar(textLine)
    numbers = sscanf(textLine, '%d ');
    line{lineCounter} = numbers;

    % Read the next line.
    textLine = fgets(fid);
    lineCounter = lineCounter + 1;
end
fclose(fid);

%findout the max number of features
maxID = 0;
for i=1:size(line,2)
    if maxID < max(line{i}(2:end))
        maxID = max(line{i}(2:end));
    end
end

%number of positive instnaces
numInstance = length(line);

fprintf('Number of data instances: %d\n',numInstance);
fprintf('Number of features: %d\n',maxID);

%create the data
TrnData = zeros(numInstance,maxID+1);
iter = 1;
for i = 1:size(line,2)
    TrnData(iter,line{i}(2:end)) = 1;
    iter = iter + 1;
end

%append the class labels
TrnData(1:numInstance,end) = class;

%save the training data matrix
csvwrite(strcat(filename,'.csv'), TrnData);
