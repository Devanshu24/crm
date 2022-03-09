function preparedata()

pos_trn_file='train.pos';
neg_trn_file='train.neg';
pos_tst_file='test.pos';
neg_tst_file='test.neg';

%%%%% Preparing Training data

%read the pos_trn file
fprintf('Reading file: %s\n',pos_trn_file)
fid = fopen(pos_trn_file);

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
numPosInstance = length(line);

%read the neg_trn file
fprintf('Reading file: %s\n',neg_trn_file)
fid = fopen(neg_trn_file);

textLine = fgets(fid); % Read first line
lineCounter = numPosInstance + 1;

while ischar(textLine)
    numbers = sscanf(textLine, '%d ');
    line{lineCounter} = numbers;

    % Read the next line.
    textLine = fgets(fid);
    lineCounter = lineCounter + 1;
end
fclose(fid);

%findout the max number of features
for i=1:size(line,2)
    if maxID < max(line{i}(2:end))
        maxID = max(line{i}(2:end));
    end
end

%total number of training instances
numTrnInstance = length(line);

fprintf('Number of data instances: %d\n',numTrnInstance);
fprintf('Number of features: %d\n',maxID);

%create the data
TrnData = zeros(numTrnInstance,maxID+1);
iter = 1;
for i = 1:size(line,2)
    TrnData(iter,line{i}(2:end)) = 1;
    iter = iter + 1;
end

%append the class labels
TrnData(1:numPosInstance,end) = 1;
TrnData(numPosInstance+1:end,end) = 0;

%save the training data matrix
csvwrite('train.csv', TrnData);

clearvars -except pos_tst_file neg_tst_file

%%%% Preparing Test Data

%read the pos_tst file
fprintf('Reading file: %s\n',pos_tst_file)
fid = fopen(pos_tst_file);

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
numPosInstance = length(line);

%read the neg_tst file
fprintf('Reading file: %s\n',neg_tst_file)
fid = fopen(neg_tst_file);

textLine = fgets(fid); % Read first line
lineCounter = numPosInstance + 1;

while ischar(textLine)
    numbers = sscanf(textLine, '%d ');
    line{lineCounter} = numbers;

    % Read the next line.
    textLine = fgets(fid);
    lineCounter = lineCounter + 1;
end
fclose(fid);

%findout the max number of features
for i=1:size(line,2)
    if maxID < max(line{i}(2:end))
        maxID = max(line{i}(2:end));
    end
end

%total number of training instances
numTstInstance = length(line);

fprintf('Number of data instances: %d\n',numTstInstance);
fprintf('Number of features: %d\n',maxID);

%create the data
TstData = zeros(numTstInstance,maxID+1);
iter = 1;
for i = 1:size(line,2)
    TstData(iter,line{i}(2:end)) = 1;
    iter = iter + 1;
end

%append the class labels
TstData(1:numPosInstance,end) = 1;
TstData(numPosInstance+1:end,end) = 0;

%save the testing data matrix
csvwrite('test.csv', TstData);
