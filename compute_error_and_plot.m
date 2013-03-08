function compute_error_and_plot(iter, Y, Y_gt, save_file, global_Y, global_Y_gt)
  mse = 0;
  % PUT YOUR CODE HERE
  % compute MSE between Y and Y_gt
  
  fprintf('MSE = %f\n', mse);
  load(save_file)
  mse_hist = [mse_hist, mse]; %#ok<NODEF>  loaded from file
  time_hist = [time_hist, cputime - starting_time];
  save(save_file, 'mse_hist', 'time_hist', 'starting_time');
  
  if exist('global_Y', 'var') && exist('global_Y_gt', 'var')
      fordisp = reshape([global_Y_gt(1:15,:)'; global_Y(1:15,:)'], 28*28, []);
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
