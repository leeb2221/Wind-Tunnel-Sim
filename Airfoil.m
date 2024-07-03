clear, clc, close all

% Simulation type:
% 1: Particle advection
% 2: Velocity Heat Map
% 3: Curl Heat Map
TYPE = 1;

% Simulation parameters
airfoil = "NACA2412";
chord = 1.5;
AOA = 10;
s = 100;                            % Grid size
ar = 5;                             % Aspect ratio
J = [0 1 0; 1 0 1; 0 1 0]/4;        % Stencil for Jacobi method
n = 200000;                         % Maximum number of particles
FPS = 30;                           % Frames per second
TPF = 1/FPS;                        % Time per frame

% Get the current jet colormap
jetMap = colormap(jet);

% Reverse the order of the colormap entries
negJetMap = flipud(jetMap);

% Create a grid
[X, Y] = meshgrid(1:s*ar, 1:s);

% Initialize pressure and velocity fields
[p, vx, vy] = deal(zeros(s, s*ar));

% Initial positions of particles
[px1, py1] = meshgrid(1:2, 2:s-1);
[px2, py2] = meshgrid(20:21, 2:s-1);
[px3, py3] = meshgrid(40:41, 2:s-1);
[px4, py4] = meshgrid(60:61, 2:s-1);
[px5, py5] = meshgrid(90:91, 2:s-1);
px = [px1, px2, px3, px4, px5];
py = [py1, py2, py3, py4, py5];
px = reshape(px, numel(px), 1);
py = reshape(py, numel(py), 1);

% Save these initial positions for the inflow
pxo = px;
pyo = py;

% Define the size of the matrix
matrix_height = 100 * chord;
matrix_width = 100 * chord;

% Generate the airfoil points and rotate them
AirfoilPoints = airfoil_generator(airfoil, 1, 60, 'closed');
x_vort = AirfoilPoints(1:2:end, 1)' * 1;
y_vort = AirfoilPoints(1:2:end, 2)' * 1;
x_vort_rotated = x_vort * cosd(-AOA) - y_vort * sind(-AOA);
y_vort_rotated = x_vort * sind(-AOA) + y_vort * cosd(-AOA);

% Scale the points to fit within the matrix size
x_scaled = round(x_vort_rotated * (matrix_width - 1) / max(x_vort)) + 30;
y_scaled = round((y_vort_rotated) * (matrix_height - 1)) + 50;

% Create the binary matrix representing the airfoil shape
airfoil_matrix = zeros(s, s*ar);
k = convhull(x_scaled, y_scaled);  % Convex hull to form the border of the airfoil
for i = 1:s
    for j = 1:s*ar
        if inpolygon(j, i, x_scaled(k), y_scaled(k))
            airfoil_matrix(i, j) = 1;
        end
    end
end

% Create a mask to apply the airfoil shape
mask = ~logical(airfoil_matrix);

% Create figure and maximize window
f = figure(1);
set(f, 'WindowState', 'maximized');

% Main simulation loop (stops when closing figure window)
while ishandle(f)
    start = tic;

    % Set initial velocity in a specific region (inflow region)
    vx(2:end-1, 2:3) = 1;
    vx(~mask) = 0;
    vy(~mask) = 0;

    % Compute right-hand side for pressure equation (Divergence of V)
    rhs = -divergence(vx, vy);

    % Jacobi iteration to solve for pressure
    for i = 1:100
        % Update interior cells using Jacobi method
        p = (conv2(p, J, 'same') + rhs/2);
        % Boundary conditions (periodic in y, reflective in x)
        p(1:1, 1:end) = p(2:2, 1:end);                    % Top boundary
        p(end:end, 1:end) = p(end-1:end-1, 1:end);        % Bottom boundary
        p(1:end, end:end) = 0;                            % Right boundary
        p(1:end, 1:1) = 1;                                % Left boundary
    end

    % Compute velocity gradient and update velocities for non-boundary pixels
    [dx, dy] = gradient(p);
    vx(2:end-1, 2:end-1) = vx(2:end-1, 2:end-1) - dx(2:end-1, 2:end-1);
    vy(2:end-1, 2:end-1) = vy(2:end-1, 2:end-1) - dy(2:end-1, 2:end-1);
    vx(~mask) = 0;
    vy(~mask) = 0;

    % Advect velocity field using Runge-Kutta 4th order method (-1 = backward)
    [pvx, pvy] = RK4(X, Y, vx, vy, -1);
    vx = interp2(vx, pvx, pvy, 'linear', 0);
    vy = interp2(vy, pvx, pvy, 'linear', 0);

    % Visualization based on simulation type
    if TYPE == 1
        % Advect particles using Runge-Kutta 4th order method (1 = forward)
        [px, py] = RK4(px, py, vx, vy, 1);

        % Add the inflow particles
        px = [px; pxo];
        py = [py; pyo];
        if length(px) > n
            px = px(end-n+1:end);
            py = py(end-n+1:end);
        end
        % Remove particles that exceed the boundary
        index = (px < s*ar-2);
        px = px(index);
        py = py(index);

        % Plot particles
        scatter(px, py, 1, 'filled');
        axis equal;
        axis([0 s*ar 0 s]);
        xticks([]);
        yticks([]);
        hold on;
        % Plot airfoil boundary
        boundary = bwboundaries(airfoil_matrix);
        for k = 1:length(boundary)
            plot(boundary{k}(:, 2), boundary{k}(:, 1), 'k', 'LineWidth', 2);
        end
        stop = toc(start);
        FPS = 1/stop;
        title(sprintf("Divergence/cell: %.5f    # Particles: %d    FPS: %.2f", ...
            sum(abs(divergence(vx(2:end-1, 10:end-10), vy(2:end-1, 10:end-10))), 'all') / (s*ar), length(px), FPS));
        hold off;
        drawnow;
    elseif TYPE == 2
        % Visualize velocity magnitude as a heat map
        velocity_Mag = sqrt(vx(2:end-1, 10:end).^2 + vy(2:end-1, 10:end).^2);
        velocity_Mag = flipud(velocity_Mag); % Flip for correct orientation
        velocity_Mag = imresize(velocity_Mag, 10, 'bicubic');
        imagesc(velocity_Mag)
        colormap(negJetMap)
        xticks([]);
        yticks([]);
        hold on;
        stop = toc(start);
        FPS = 1/stop;
        title(sprintf("Divergence/cell: %.5f FPS: %.2f", ...
            sum(abs(divergence(vx(2:end-1, 10:end-10), vy(2:end-1, 10:end-10))), 'all') / (s*ar), FPS));
        axis equal;
        hold off;
        drawnow;
    elseif TYPE == 3
        % Visualize Curl field as a heat map
        CURL = abs(curl(vx, vy));
        CURL = CURL(3:end-2, 10:end-10); % Exclude boundary for clearer view
        CURL = flipud(CURL); % Flip for correct orientation
        CURL = imresize(CURL, 10, 'bicubic');
        imagesc(CURL)
        colormap(negJetMap)
        xticks([]);
        yticks([]);
        axis off
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
