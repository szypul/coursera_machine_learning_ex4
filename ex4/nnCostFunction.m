function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m,1),X];
z2 = Theta1*a1';
a2= [ones(1,m); sigmoid(z2)];
z3 = Theta2*a2;
h = sigmoid(z3); %a3

Y = bsxfun(@eq, y,1:num_labels);

J = sum(sum(( ( -1*Y).*log(h)' - (1-Y).*log(1-h)'  ),2))/m + (lambda/(2*m)) * ( sum(sum((Theta1(:,2:end)).^2,2)) + sum(sum((Theta2(:,2:end)).^2,2)));

delta3 = h'-Y;
delta2 = Theta2(:,2:end)'*delta3'.*sigmoidGradient(z2);

Delta2 = delta3'*a2';
Delta1 = delta2*a1;


%grad = Delta/m;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

%grad = [Theta1_grad(:) ; Theta2_grad(:)];



% This works (without regularization
%grad = [Delta1(:)/m ; Delta2(:)/m];
% it worked! do not delte it.

Reg_Delta1 = Delta1 + lambda*[zeros(hidden_layer_size, 1), Theta1(:,2:end)];
Reg_Delta2 = Delta2 + lambda*[zeros(num_labels,1), Theta2(:,2:end)];


grad = [Reg_Delta1(:)/m; Reg_Delta2(:)/m];



end
