clear; clc; close all;

% Simulation type: 
% 1 - Particle Advection
% 2 - Velocity Heat Map, 
% 3 - Curl Heat Map
TYPE = 3;

% Simulation parameters
s = 100;        % Grid size
ar = 5;         % Aspect ratio
J = [0 1 0; 1 0 1; 0 1 0]/4;  % Jacobi matrix for pressure solver
n = 120000;     % Maximum number of particles

% Initialize colormap
jetMap = colormap(jet);
negJetMap = flipud(jetMap);

% Define cube vertices
a = 40;
b = 60;
c = 50;
d = 55;

% Create grid
[X, Y] = meshgrid(1:s*ar, 1:s);

% Initialize pressure and velocity fields
p = zeros(s, s*ar);
vx = zeros(s, s*ar);
vy = zeros(s, s*ar);

% Initial positions of particles
[px, py] = meshgrid(10:10, 1:s);

% Store initial positions for inflow
pxo = px;
pyo = py;

% Set up figure for visualization
f = figure(1);
set(f, 'WindowState', 'maximized');

% Main simulation loop (stops when closing figure window)
while ishandle(f)
    start = tic;

    % Set boundary conditions and velocity modifications
    vx(a:b, c:d) = 0;   % Zero velocity inside cube
    vy(a:b, c:d) = 0;   % Zero velocity inside cube

    % Compute right-hand side for pressure equation (Divergence of V)
    rhs = divergence(vx, vy)/2;

    % Jacobi iteration to solve for pressure
    for i = 1:100
        p = (conv2(p, J, 'same') - rhs);
        % Boundary conditions for pressure
        p(1:1,1:end) = p(2:2,1:end);                % Top Bound
        p(end:end,1:end) = p(end-1:end-1,1:end);    % Bottom Bound
        p(1:end,end-5:end) = 0;                     % Side Bound
        p(1:end,1:5) = 1;                          % Side Bound
    end

    % Compute velocity gradient and update velocities
    [dx, dy] = gradient(p);
    vx(2:end-1, 2:end-1) = vx(2:end-1, 2:end-1) - dx(2:end-1, 2:end-1);
    vy(2:end-1, 2:end-1) = vy(2:end-1, 2:end-1) - dy(2:end-1, 2:end-1);

    % Advect velocity field using Runge-Kutta 4th order method
    [pvx, pvy] = RK4(X, Y, vx, vy, -1);
    vx = interp2(vx, pvx, pvy, 'linear', 0);
    vy = interp2(vy, pvx, pvy, 'linear', 0);

    % Visualization based on simulation type
    if TYPE == 1
        % Advect particles using Runge-Kutta 4th order method
        [px, py] = RK4(px, py, vx, vy, 1);

        % Add inflow particles
        px = [px; pxo];
        py = [py; pyo];

        % Remove excess particles
        if length(px) > n
            px = px(end-n+1:end);
            py = py(end-n+1:end);
        end

        % Remove outflow particles and those inside the cube
        index = (px < s*ar-2) & ~((px >= c) & (px <= d) & (py >= a) & (py <= b));
        px = px(index);
        py = py(index);

        % Plot particles
        scatter(px, py, 1, 'filled');
        axis equal;
        axis([0 s*ar 0 s]);
        xticks([]);
        yticks([]);
        hold on;
        % Highlight cube region
        rectangle('Position', [c, a, d-c, b-a], 'EdgeColor', 'r', 'LineWidth', 2);
        stop = toc(start);
        FPS = 1/stop;
        % Update title with simulation metrics
        title(sprintf("DIV/cell: %.5f    # Particles: %d    FPS: %.2f", ...
            sum(abs(divergence(vx(2:end-1,10:end-10), vy(2:end-1,10:end-10))), 'all')/(s*ar), length(px), FPS));
        hold off;
        drawnow;
        
    elseif TYPE == 2
        % Visualize velocity magnitude
        velocity_Mag = sqrt(vx(2:end-1,10:end).^2 + vy(2:end-1,10:end).^2);
        velocity_Mag = imresize(velocity_Mag, 10, 'bicubic');
        imagesc(velocity_Mag);
        colormap(negJetMap);
        xticks([]);
        yticks([]);
        axis equal;
        stop = toc(start);
        FPS = 1/stop;
        title(sprintf("DIV/cell: %.5f FPS: %.2f", ...
            sum(abs(divergence(vx(2:end-1,10:end-10), vy(2:end-1,10:end-10))), 'all')/(s*ar), FPS));
        drawnow;

    elseif TYPE == 3
        % Visualize curl field
        CURL = abs(curl(vx, vy));
        CURL = CURL(2:end-1,10:end-10);
        CURL = imresize(CURL, 10, 'bicubic');
        imagesc(CURL);
        colormap(negJetMap);
        xticks([]);
        yticks([]);
        axis equal;
        drawnow;
    end
end

% Function for Runge-Kutta 4th order method for advection
function [x_new, y_new] = RK4(px, py, vx, vy, h)
   k1x = interp2(vx, px, py, 'linear', 0);
   k1y = interp2(vy, px, py, 'linear', 0);
   k2x = interp2(vx, px + h/2 * k1x, py + h/2 * k1y, 'linear', 0);
   k2y = interp2(vy, px + h/2 * k1x, py + h/2 * k1y, 'linear', 0);
   k3x = interp2(vx, px + h/2 * k2x, py + h/2 * k2y, 'linear', 0);
   k3y = interp2(vy, px + h/2 * k2x, py + h/2 * k2y, 'linear', 0);
   k4x = interp2(vx, px + h * k3x, py + h * k3y, 'linear', 0);
   k4y = interp2(vy, px + h * k3x, py + h * k3y, 'linear', 0);
   x_new = px + h/6 * (k1x + 2*k2x + 2*k3x + k4x);
   y_new = py + h/6 * (k1y + 2*k2y + 2*k3y + k4y);
end
