function compute_error_and_plot(iter, Y, Y_gt, do_plot)  
  mse = 0;
  % PUT YOUR CODE HERE
  % compute MSE between Y and Y_gt
  
  fprintf('MSE = %f\n', mse);
  load pretrain_err
  mse_hist = [mse_hist, mse]; %#ok<NODEF>  loaded from file
  save pretrain_err mse_hist
  
  if ~exist('do_plot', 'var') || do_plot
      fordisp = reshape([Y_gt(1:15,:)'; Y(1:15,:)'], 28*28, []);
      if iter == 1 
        close all 
        figure('Position',[100,600,1000,200]);
      else 
        figure(1)
      end 
      mnistdisp(fordisp);
      drawnow;
  end
end
