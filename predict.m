function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
hx=sigmoid(X*Theta1');
hx = [ones(m, 1) hx];
hx = sigmoid(hx*Theta2');
[tmp p] = max(hx,[],2);    

end
