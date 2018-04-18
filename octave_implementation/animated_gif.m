function plot(N, dt, t_final, visc)
[u, v, number_of_steps, t_steady] = lid_driven_cavity(N, dt, t_final, visc);

velocity_tile = sqrt(u.^2 + v.^2);
x_linspace    = linspace(0, 1, N);
x_tile        = repmat(x_linspace, N, 1);
y_tile        = transpose(x_tile);


h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'n_100.gif';
for n = 1:10:(t_steady/dt)
    % Draw plot for y = x.^n
    speed_tile = velocity_tile(:, :, n);
    contourf(x_tile, y_tile, speed_tile, 'edgecolor', 'none');
    colorbar;
    title(n*dt);
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
  end
disp(number_of_steps);
end
