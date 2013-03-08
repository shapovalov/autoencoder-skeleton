DO_PRETRAIN = true;

digitdata=[]; 
for i = 0:9
  load(['digit' num2str(i)]); 
  digitdata = [digitdata; D];  %#ok<AGROW> only 10 times
end

nobjects = size(digitdata,1);
digitdata = digitdata(randperm(nobjects),:);  

arch1.actfun = @(X) ones(size(X));   % activation function f(X)
arch1.dactfun = @(Y) zeros(size(Y)); % its derivative AS A FUNCTION OF Y, Y=f(X)
% PUT YOUR CODE HERE
% describe logistic sigmoid and its derivative

arch(1:9) = arch1;
[arch.numw] = deal(784,1000,500,250,30,250,500,1000,784);

% override activation function for the middle layer
arch(5).actfun = @(X) ones(size(X));   % activation function f(X)
arch(5).dactfun = @(Y) zeros(size(Y)); % its derivative AS A FUNCTION OF Y, Y=f(X)
% PUT YOUR CODE HERE
% describe logistic sigmoid and its derivative


if DO_PRETRAIN
  tic; 
  [w, b] = pretrain(arch, digitdata);
  toc
  wflat = flatten_weights(w, b);
else
  numw = [arch.numw]; %#ok<UNRCH>
  nweights = sum((numw(1:end-1)+1) .* numw(2:end));
  wflat = zeros(nweights, 1);
  % PUT YOUR CODE HERE
  % replace with random gaussian noise when try to train without pretraining
end

save('pretrain_weights.mat', 'wflat', 'arch');


% mse_hist = [];
% time_hist = [];
% starting_time = cputime;
% save('train_err.mat', 'mse_hist', 'time_hist', 'starting_time');

% tic;
% wflat = minimize(wflat, digitdata, digitdata, arch, false, 30, 1500, ...
%       @(iter, Y) compute_error_and_plot(iter, Y, digitdata, 'train_err.mat', Y, digitdata), 0.97);
% toc

% save('final_weights.mat', 'wflat', 'arch');
