%desired values 1x1707, 1707 number
xtest_target = load("dzip.mat");
test_target = xtest_target.dzip;

%drawings 256x1707   
xtest_input = load("azip.mat");
test_input = xtest_input.azip;


%test desired value 1x2707
xtrain_target = load("dtest.mat");
train_target = xtrain_target.dtest

%test drawing value 256x2707
xtrain_input = load("testzip.mat");
train_input = xtrain_input.testzip;


train_target = transition(10, train_target); %one-hot encoded 
test_target = transition(10, test_target);   %makes it easier to train the nw which class is correct


% net = feedforwardnet([30, 30]);
% net = configure(net, train_input, train_target);

net= patternnet(30);

[net, tr] = train(net, train_input, train_target);

output = net(test_input);
errors = test_target - output;

%test

% Extract the first 10 test samples
num_samples_to_visualize = 10;
selected_test_samples = test_input(:, 1:num_samples_to_visualize);

% Loop through the selected samples and visualize using ima2()
for i = 1:num_samples_to_visualize
    sample = selected_test_samples(:, i); % Extract a single sample
    ima2(sample); % Call ima2 function to visualize the sample
    title(['Test Sample ', num2str(i)]); % Add a title to the plot
    pause(0.9);
end

%test



performance = perform(net, test_target, output);

[~, max_indices] = max(output, [], 1); 

custom_output = zeros(size(output));

for i = 1:size(custom_output, 2)
    custom_output(max_indices(i), i) = 1;   % Set the max row in each column to 1
end

toplam = zeros(10,1);

for i = 1:size(custom_output, 2)
    toplam = toplam + custom_output(:, i);
end

realSum = zeros(10,1);

for i = 1:size(test_target, 2)
    realSum = realSum + test_target(:, i);
    endn

% Initialize confusion matrix
num_classes = 10;  % Classes 0-9
confMat = zeros(num_classes, num_classes);

% Generate class labels from counts
actual_labels = [];
predicted_labels = [];

for i = 1:num_classes
    actual_labels = [actual_labels, repmat(i, 1, realSum(i))];  % Actual
    predicted_labels = [predicted_labels, repmat(i, 1, toplam(i))];  % Predicted
end

% Now populate confusion matrix
for i = 1:length(actual_labels)
    actual = actual_labels(i);
    predicted = predicted_labels(i);
    
    % Increment appropriate cell in confusion matrix
    confMat(actual, predicted) = confMat(actual, predicted) + 1;
end

% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Optional: Heatmap Visualization
figure;
heatmap(0:9, 0:9, confMat, 'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted Class', 'YLabel', 'Actual Class');




total_samples = sum(confMat(:));  % Total number of samples
correct_predictions = sum(diag(confMat));  % Sum of diagonal elements
accuracy = correct_predictions / total_samples;

disp(['Accuracy: ', num2str(accuracy * 100), '%']);


precision = zeros(1, size(confMat, 1));
for i = 1:size(confMat, 1)
    precision(i) = (confMat(i, i) / sum(confMat(:, i))) *100;  % TP / (TP + FP)
end

disp('Precision for each class:');
disp(precision);


recall = zeros(1, size(confMat, 1));
for i = 1:size(confMat, 1)
    recall(i) = (confMat(i, i) / sum(confMat(i, :)))*100;  % TP / (TP + FN)
end

disp('Recall for each class:');
disp(recall);


f1_score = zeros(1, size(confMat, 1));
for i = 1:size(confMat, 1)
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

disp('F1-Score for each class:');
disp(f1_score);


macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1_score = mean(f1_score);

disp(['Macro Precision: ', num2str(macro_precision)]);
disp(['Macro Recall: ', num2str(macro_recall)]);
disp(['Macro F1-Score: ', num2str(macro_f1_score)]);


% Assuming precision, recall, f1_score, and accuracy are already calculated
% and are vectors with a value for each class.

% Define class labels
class_labels = arrayfun(@(x) ['Class ', num2str(x)], 0:num_classes-1, 'UniformOutput', false);

accuracy =  [accuracy, zeros(1,9)];



% Initialize per-class accuracy
per_class_accuracy = zeros(1, num_classes);

% Total number of samples
total_samples = sum(confMat(:));

for i = 1:num_classes
    % True Positives for class i
    TP = confMat(i, i);
    
    % True Negatives for class i (sum of correct predictions for other classes)
    TN = total_samples - sum(confMat(i, :)) - sum(confMat(:, i)) + TP;
    
    % Per-class accuracy calculation
    per_class_accuracy(i) = ((TP + TN) / total_samples)*100;
end

disp('Per-Class Accuracy:');
disp(per_class_accuracy);



% Create table
T = table(class_labels', precision', recall', f1_score', per_class_accuracy', ...
          'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score', 'Accuracy'});

% Display the table
disp('Classification Metrics Table:');
disp(T);



function []=ima2(A)
% Translate vector to become nonnegative
% Scale to interval [0,20]
% Reshape the vector as a matrix and then show image

a1=squeeze(A);  
a1=reshape(a1,16,16)';  

a1=(a1-min(min(a1))*ones(size(a1)));
a1=(20/max(max(a1)))*a1;

mymap1 =[1.0000    1.0000    1.0000
    0.8715    0.9028    0.9028
    0.7431    0.8056    0.8056
    0.6146    0.7083    0.7083
    0.4861    0.6111    0.6111
    0.3889    0.4722    0.5139
    0.2917    0.3333    0.4167
    0.1944    0.1944    0.3194
    0.0972    0.0972    0.1806
         0         0    0.0417];
colormap(mymap1)


image(a1)
end