function [u_out, v_out, dsolution, t_steady] = lid_driven_cavity(N, dt, t_final, visc)

number_of_steps = floor(t_final / dt);

t_steady = t_final;

u = zeros((N-2)^2, 1);
v = zeros((N-2)^2, 1);

n_square = length(u);
n        = sqrt(n_square);

u_out = zeros(N, N, number_of_steps);
v_out = zeros(N, N, number_of_steps);

delta_x = 1 / (N-1);

% Matrices for projection and diffusion created using spdiag.

% Off diagonal matrix elements.
diagonal_n = -1*ones(n_square,1);
diagonal_plus_one = -1*ones(n_square,1);
diagonal_plus_one(n+1:n:end) = 0;

diagonal_minus_one = -1*ones(n_square,1);
diagonal_minus_one(sqrt(n_square):sqrt(n_square):end) = 0;

diagonal = (4 + delta_x^2/(dt*visc))*ones(n_square,1);

A_diffuse = (dt*visc/delta_x^2)*spdiags([diagonal_n, diagonal_minus_one,...
             diagonal, diagonal_plus_one, diagonal_n], [-sqrt(n_square),...
             -1, 0 1, sqrt(n_square)], n_square, n_square);

n_square = N^2;
diagonal_plus_one = 1*ones(n_square,1);
diagonal_plus_one(sqrt(n_square)+1:sqrt(n_square):end) = 0;
diagonal_plus_one(2:sqrt(n_square):end) = 2;

diagonal_plus_n = 1*ones(n_square,1);
diagonal_plus_n(1:sqrt(n_square)+sqrt(n_square)) = 2;

diagonal_minus_one = 1*ones(n_square,1);
diagonal_minus_one(sqrt(n_square):sqrt(n_square):end) = 0;
diagonal_minus_one(sqrt(n_square)-1:sqrt(n_square):end) = 2;

diagonal_minus_n = 1*ones(n_square,1);
diagonal_minus_n(n_square-2*sqrt(n_square)+1:n_square) = 2;

diagonal = -4*ones(n_square,1);

A_project = 1/(delta_x^2)*spdiags([diagonal_minus_n, diagonal_minus_one,...
            diagonal, diagonal_plus_one, diagonal_plus_n],...
            [-sqrt(n_square), -1, 0 1, sqrt(n_square)],...
            n_square, n_square);

A_project(end, :)   = 0;
A_project(end, end) = 1;

%enter time loop
t = 0;
tstep = 0;

u_old = u;
v_old = v;

% Set tolerance for steady state.
tol = 1e-8;

while t < t_final
    t = t+dt;
    disp(t);
    tstep = tstep + 1;

    [u,v] = diffuse(u, v, delta_x, dt, visc, A_diffuse);
    [u,v] = advect(u,v,delta_x,dt);
    [u,v] = project(u,v,delta_x, A_project);

    dsolution = norm(([u,;v]-[u_old;v_old])/N^2);
    if (dsolution < tol)
        disp(['steady state reached at t=',num2str(t)]);
        t_steady = t;
        break;
    end

    u_old = u;
    v_old = v;

    [utmp,vtmp] = add_bc(u,v);

    u_out(:,:,tstep) = rot90(utmp);
    v_out(:,:,tstep) = rot90(vtmp);
end

% Contour plot at the final timestep.
%x_linspace = linspace(0, 1, sqrt(n_square));
%x_tile     = repmat(x_linspace, sqrt(n_square), 1);
%y_tile     = transpose(x_tile);
%speed_tile = sqrt(u_out(:, :, end).^ 2 + v_out(:, :, end).^ 2);
%figure
%contourf(x_tile, y_tile, speed_tile,'edgecolor', 'none');

end


% diffusion step - solve diffusion equation using backwards Euler
function [u_new, v_new] = diffuse(u, v, delta_x, dt, visc, A_diffusion)

n_square = length(u);
u(1:sqrt(n_square)) = u(1:sqrt(n_square)) + dt*visc/delta_x^2;

% mldivide is much faster than passing inverse of matrices.
u_new = A_diffusion \ u;
v_new = A_diffusion \ v;

end


%% projection step - solve Poisson equation
function [u_new, v_new] = project(u, v, delta_x, A_projection)

n_square = (sqrt(length(u)) + 2)^2;
n        = sqrt(n_square);
[u_w_bc, v_w_bc] = add_bc(u,v);

%calculate div(velocity)
divergence = zeros(n,n);

%vectoized divergence interior
divergence(2:n-1, 2:n-1)  = (u_w_bc(1:n-2, 2:n-1)-u_w_bc(3:n, 2:n-1))...
                          / (2 * delta_x)...
                          + (v_w_bc(2:n-1, 1:n-2)- v_w_bc(2:n-1, 3:n))...
                          / (2 * delta_x);

%Vectorized boundary conditions
divergence(1, 2:n-1)   = -u_w_bc(2, 2:n-1)    / delta_x;
divergence(end, 2:n-1) = u_w_bc(end-1, 2:n-1) / delta_x;
divergence(2:n-1, 1)   = -v_w_bc(2:n-1, 2)    / delta_x;
divergence(2:n-1, end) = v_w_bc(2:n-1, end-1) / delta_x;

divergence = reshape(divergence, n_square, 1);

%solve poisson equation for q
q = A_projection \ divergence;

%calculate grad(q)
q = reshape(q, n, n);

qx = zeros(n,n);
qy = zeros(n,n);


qx(2:n-1, 2:n-1) = (q(1:n-2, 2:n-1)-q(3:n, 2:n-1)) / (2*delta_x);

qy(2:n-1, 2:n-1) = (q(2:n-1, 1:n-2)-q(2:n-1, 3:n)) / (2*delta_x);

%left and right boundaries, qx = 0 from boundary conditions


qy(1, 2:n-1) = (q(1, 1:n-2)-q(1, 3:n)) / (2 * delta_x);

qy(end, 2:n-1) = (q(end, 1:n-2)-q(end, 3:n)) / (2 * delta_x);
%top and bottom boundaries, qy = 0 from boundary conditions


qx(2:n-1, 1) = (q(1:n-2, 1)-q(3:n, 1)) / (2*delta_x);


qx(2:n-1, end) = (q(1:n-2, end)-q(3:n, 1)) / (2*delta_x);

u_new = u_w_bc - qx;
v_new = v_w_bc - qy;

u_new = reshape(u_new(2:end-1,2:end-1), size(u));
v_new = reshape(v_new(2:end-1,2:end-1), size(v));

end

% advection step
function [u_new_flat, v_new_flat] = advect(u, v, delta_x, dt)
n_minus_2 = sqrt(length(u));
n         = n_minus_2 + 2;

[u_w_bc, v_w_bc] = add_bc(u,v);

%vectorized bilinear interpolation
u_tile = reshape(u, n_minus_2, n_minus_2);
v_tile = reshape(v, n_minus_2, n_minus_2);

y0_linspace  = linspace(0, 1, n);
y0_tile_temp = repmat(y0_linspace, n, 1);
y0_tile      = y0_tile_temp(2:end-1, 2:end-1);
x0_tile      = transpose(y0_tile);


zero_tile = zeros(size(u_tile));
one_tile = ones(size(u_tile));

x1_tile = min(max(x0_tile - u_tile * dt, zero_tile), one_tile);
y1_tile = min(max(y0_tile - v_tile * dt, zero_tile), one_tile);

i_left_tile  = transpose(min(floor(x1_tile / delta_x) + 1,...
                            (n_minus_2 + 1) * one_tile));
i_right_tile = i_left_tile + 1;
i_left_flat  = reshape(i_left_tile, n_minus_2 * n_minus_2, 1);
i_right_flat = reshape(i_right_tile, n_minus_2 * n_minus_2, 1);


j_bottom_tile = transpose(min(floor(y1_tile / delta_x) + 1,...
                              (n_minus_2 + 1) * one_tile));
j_top_tile    = j_bottom_tile + 1;
j_bottom_flat = reshape((j_bottom_tile - 1) * n, n_minus_2 * n_minus_2, 1);
j_top_flat    = reshape((j_top_tile - 1) * n, n_minus_2 * n_minus_2, 1);

x_left_tile  = (i_left_tile - 1) / (n - 1);
x_right_tile = (i_right_tile - 1) / (n - 1);

y_bottom_tile = (j_bottom_tile - 1) / (n - 1);
y_top_tile    = (j_top_tile - 1) / (n - 1);

flat_u = reshape((u_w_bc), n * n, 1);
flat_v = reshape((v_w_bc), n * n, 1);


u_top_left     = reshape(flat_u(i_left_flat  + j_top_flat),...
                                n_minus_2, n_minus_2);
u_top_right    = reshape(flat_u(i_right_flat + j_top_flat),...
                                n_minus_2, n_minus_2);
u_bottom_right = reshape(flat_u(i_right_flat + j_bottom_flat),...
                                n_minus_2, n_minus_2);
u_bottom_left  = reshape(flat_u(i_left_flat  + j_bottom_flat),...
                                n_minus_2, n_minus_2);

v_top_left     = reshape(flat_v(i_left_flat  + j_top_flat),...
                                n_minus_2, n_minus_2);
v_top_right    = reshape(flat_v(i_right_flat + j_top_flat),...
                                n_minus_2, n_minus_2);
v_bottom_right = reshape(flat_v(i_right_flat + j_bottom_flat),...
                                n_minus_2, n_minus_2);
v_bottom_left  = reshape(flat_v(i_left_flat  + j_bottom_flat),...
                                n_minus_2, n_minus_2);

x1_tile = transpose(x1_tile);
y1_tile = transpose(y1_tile);


u_x_interpolation_lower = ((x_right_tile - x1_tile).* u_bottom_left ...
                        + (x1_tile - x_left_tile).*u_bottom_right)/delta_x;

u_x_interpolation_upper = ((x_right_tile - x1_tile).* u_top_left ...
                        + (x1_tile - x_left_tile).*u_top_right)/delta_x;

u_new_tile = ((y_top_tile - y1_tile).* u_x_interpolation_lower ...
           + (y1_tile - y_bottom_tile).* u_x_interpolation_upper)/delta_x;

v_x_interpolation_lower = ((x_right_tile - x1_tile).* v_bottom_left ...
                        + (x1_tile - x_left_tile).*v_bottom_right)/delta_x;

v_x_interpolation_upper = ((x_right_tile - x1_tile).* v_top_left ...
                        + (x1_tile - x_left_tile).* v_top_right) / delta_x;

v_new_tile = ((y_top_tile - y1_tile).* v_x_interpolation_lower ...
           + (y1_tile - y_bottom_tile).* v_x_interpolation_upper)/delta_x;

u_new_tile = transpose(u_new_tile);
v_new_tile = transpose(v_new_tile);

u_new_flat = reshape(u_new_tile, n_minus_2 * n_minus_2, 1);
v_new_flat = reshape(v_new_tile, n_minus_2 * n_minus_2, 1);

end


function [u_w_bc, v_w_bc] = add_bc(u,v)

n_square = length(u);
n = sqrt(n_square);

u = reshape(u, n, n);
v = reshape(v, n, n);

u_w_bc = zeros(n + 2, n + 2);
v_w_bc = zeros(n + 2, n + 2);

u_w_bc(:,1) = 1;
u_w_bc(2:end-1,2:end-1) = u;

v_w_bc(2:end-1,2:end-1) = v;

end
