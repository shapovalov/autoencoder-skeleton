load digit0       % load only digit 0 instances
data = D;

arch1.actfun = @(X) ones(size(X));   % activation function f(X)
arch1.dactfun = @(Y) zeros(size(Y)); % its derivative AS A FUNCTION OF Y, Y=f(X)
% PUT YOUR CODE HERE
% describe logistic sigmoid and its derivative

clear arch
arch1.numw = 0;
arch(1:3) = arch1;
[arch.numw] = deal(784,1000,784);

numw = [arch.numw];
% weights are tied, so W matrices are stored for only the first half of layers
nweights = sum((numw(1:end-1)+1) .* numw(2:end)) - ...
  sum((numw(1:end-1)) .* numw(2:end)) / 2;

% zero initialization. Try random gaussian instead
wflat = zeros(nweights, 1);

% this is for logging all error computations. See compute_error_and_plot function.
mse_hist = [];
time_hist = [];
starting_time = cputime;
save('pretrain_err.mat', 'mse_hist', 'time_hist', 'starting_time');

nepochs = 20;
nbatches = 1;
momentum = 0.0;
iter_callback = @(iter, Y) compute_error_and_plot(iter, Y, data, 'pretrain_err.mat');

tic;
wflat = minimize(wflat, data, data, arch, true, nepochs, true, nbatches, ...
  iter_callback, momentum);
toc
