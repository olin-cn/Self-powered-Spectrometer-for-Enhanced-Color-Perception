% --------------------- 
% @Description: Spectrum Reconstruction for vdWH (Simplified)
% @Version: 2.1
% @Author: Xiangfu Lei, Hanxiao Cui
% @Date: 2026-05-06 
% --------------------- 
clearvars; clc;

%% 1. Load Model Kernel
% Read kernel matrix from Excel file
K1 = table2array(readtable('RMatrix_1.xlsx'));
Kernel = K1(2:end, 2:end) / max(max(K1(2:end, 2:end)));
Wavelength = K1(2:end, 1);

% Interpolate to 601 wavelength points for finer spectral resolution
Kernel = expand(Kernel, 601);
Wavelength = expand(Wavelength, 601);

% Model parameters
NumSensors = size(Kernel, 2);
NumDataPoints = length(Wavelength);
WavelengthRange = [Wavelength(1), Wavelength(end)];
dWavelength = WavelengthRange(2) - WavelengthRange(1);

%% 2. Visualize Model Response (Kernel Matrix)
figure(1); clf;
pcolor(Wavelength, 1:NumSensors, Kernel'); hold on
shading('flat'); colormap('jet'); colorbar;
ylabel('Sensor No.');
pbaspect([1 1 1]);
set(gca, 'FontSize', 12);
set(gcf, 'Color', 'white');

%% 3. Generate Ground Truth Spectra
% Create 6 single-peak Gaussian spectra at different center wavelengths (400:100:900 nm)
NumSpectra = 6;
CentresWL = 400:100:900;                         % Center wavelengths in nm
Centres = (CentresWL - 360) / (1000 - 360);      % Normalize to [0, 1] (360-1000 nm range)

% Wavelength normalization helpers
fNormalizeWL = @(x) (x - WavelengthRange(1)) / dWavelength;
fInvNormalizeWL = @(y) y * dWavelength + WavelengthRange(1);

% Generate each spectrum
Data = struct('Wavelength', [], 'Spectrum', [], 'Peaks', []);
for i = 1:NumSpectra
    Data(i) = MakeSpectrumData(fNormalizeWL(Wavelength), MakeSpectrumConfig(Centres(i)));
    Data(i).Wavelength = fInvNormalizeWL(Data(i).Wavelength);
end

% Plot ground truth spectra
figure(2); clf;
for i = 1:NumSpectra
    plot(Data(i).Wavelength, Data(i).Spectrum, 'LineWidth', 1.5); hold on
end
xlabel('Wavelength (nm)'); ylabel('Normalized Intensity');
grid on; pbaspect([1 1 1]);
set(gca, 'FontSize', 12); set(gcf, 'Color', 'white');
legend(arrayfun(@(c) sprintf('%.0f nm', c), CentresWL, 'UniformOutput', false), ...
    'Location', 'best');

%% 4. Reconstruction from Experimental Data (360-1000 nm narrowband)
load('narrow360-1000.mat');  % Pre-computed measurement responses

% Solver configuration
SolverConfig.nonnegative = true;    % Enforce non-negative spectrum
SolverConfig.regularized = true;    % Enable regularization framework
SolverConfig.intrule = 'trapz';     % Trapezoidal integration rule
SolverConfig.alpha = 0;             % L1 penalty coefficient (disabled)
SolverConfig.beta = [0, 0];         % L2 penalty coefficients [0th-order, 1st-order] (disabled)

% Reconstruct each spectrum
Solution = struct('Wavelength', [], 'Spectrum', []);
for i = 1:NumSpectra
    % Measurement data: columns 5, 15, 25, 35, 45, 55 of the narrow matrix
    measurement = narrow(:, 5 + 10*(i-1))';

    % Perform reconstruction via non-negative least squares
    Solution(i) = GetSpectraData(Wavelength, Kernel, SolverConfig, measurement);

    % Visualization: reconstruction vs ground truth
    % figure(2 + i); clf;
    ShowFigure(Solution(i), Wavelength, Data(i));hold on
end

fprintf('Reconstruction complete. %d spectra processed.\n', NumSpectra);


%% ====================== Core Functions ======================

% --------------------- 
% @Description: Reconstruct spectrum from measurement response
% --------------------- 
function Solution = GetSpectraData(Wavelength, Kernel, SolverConfig, measurement)
    % Solve the inverse problem: measurement = K * spectrum
    [ti, xi] = SolveSpectrumNaive(Wavelength, Kernel, measurement, SolverConfig);

    % Store solution
    Solution.Wavelength = ti;
    Solution.Spectrum = xi / max(xi(:));
end

% --------------------- 
% @Description: Generate ground-truth Gaussian spectrum data
% --------------------- 
function Data = MakeSpectrumData(Wavelength, config)
    % Default weights = 1 if not specified
    if ~isfield(config, 'weights')
        config.weights = ones(size(config.fwhm));
    end

    % Input validation
    assert(isvector(config.centres) && isvector(config.fwhm), ...
        'centres and fwhm must be vectors');
    assert(length(config.centres) == length(config.fwhm), ...
        'Number of centres and fwhm must match');
    assert(length(config.centres) == length(config.weights), ...
        'Number of centres and weights must match');

    % Superposition of Gaussian peaks
    Spectrum = zeros(size(Wavelength));
    for i = 1:length(config.centres)
        Spectrum = Spectrum + config.weights(i) * GetGaussian(Wavelength, config.centres(i), config.fwhm(i));
    end

    Data.Spectrum = Spectrum / max(Spectrum);
    Data.Wavelength = Wavelength;
    Data.Peaks = config.centres;

    % Nested: single Gaussian peak function
    function Sp = GetGaussian(xi, mu, fwhm)
        sigma = fwhm / (2 * sqrt(2 * log(2)));  % Convert FWHM to standard deviation
        Sp = exp(-0.5 * ((xi - mu) / sigma).^2);
    end
end

% --------------------- 
% @Description: Configure spectrum parameters (centres, FWHM, weights)
% --------------------- 
function Config = MakeSpectrumConfig(Centres)
    Config.centres = Centres;
    Config.fwhm    = 0.003;   % Peak width (normalized units)
    Config.weights = 1;       % Peak amplitude weight
end

% --------------------- 
% @Description: Solve spectrum reconstruction via non-negative least squares
%   min 0.5 * x'*H*x + f'*x   subject to x >= 0
%   where H = A'*A, f = -A'*b
% --------------------- 
function [vi, spectrum, config] = SolveSpectrumNaive(wavelength, kernel, response, config)
    % Input validation
    assert(iscolumn(wavelength), 'Wavelength must be a column vector');
    assert(all(diff(wavelength) > 0), 'Wavelength must be strictly increasing');
    assert(length(wavelength) == size(kernel, 1), ...
        'Wavelength length must match kernel rows');
    assert(size(response, 2) == size(kernel, 2), ...
        'Number of responses must match kernel columns');

    % Build coefficient matrix A via trapezoidal integration
    [A, ti] = IntRuleTrapz(wavelength, kernel);
    b = response';

    % Form normal equations: H*x = f  →  (A'*A)*x = A'*b
    H = A' * A;
    f = -A' * b;

    % Solve the quadratic programming problem
    [xi, config] = SolveLinearForm(H, f, config);

    % Re-sample solution to original wavelength grid
    vi = linspace(ti(1), ti(end), length(wavelength))';
    spectrum = interp1(ti, xi, vi, 'pchip');
end

% --------------------- 
% @Description: Trapezoidal integration rule for building coefficient matrix A
%   A(i,j) = ∫ kernel(λ, i) dλ ≈ Σ w_k * kernel(λ_k, i)
% --------------------- 
function [A, wl] = IntRuleTrapz(wl, kernel)
    % Trapezoidal weights for numerical integration
    weights = [0; 0.5 * diff(wl)];              % Leading half-step
    weights = weights + circshift(weights, -1);  % Add trailing half-step
    A = (repmat(weights, 1, size(kernel, 2)) .* kernel)';
end

% --------------------- 
% @Description: Solve min 0.5*x'*H*x + f'*x with optional regularization
%   Uses MATLAB quadprog (interior-point-convex solver)
% --------------------- 
function [xi, config] = SolveLinearForm(H, f, config)
    % Apply Tikhonov regularization if enabled
    if config.regularized
        % Extract penalty parameters
        alpha = config.alpha;
        beta = config.beta;

        % Regularize the normal equations
        [H, f] = Regularize(H, f, alpha, beta);
    end

    % Quadratic programming options (suppress display)
    options = optimoptions(@quadprog, 'Display', 'off');

    % Solve: min 0.5*x'*H*x + f'*x
    if config.nonnegative
        % Non-negativity constraint: x_i >= 0
        y = quadprog(H, f, [], [], [], [], zeros(size(H, 2), 1), [], [], options);
    else
        % Unconstrained
        y = quadprog(H, f, [], [], [], [], [], [], [], options);
    end

    config.BasisCoefficient = y;
    xi = y;  % Direct solution (trapezoidal rule uses point-wise basis)
end

% --------------------- 
% @Description: Apply Tikhonov regularization to normal equations
%   H_reg = H + beta_0*I + beta_1*L1'*L1 + ...
%   f_reg = f + alpha
% --------------------- 
function [Q, P] = Regularize(H, f, alpha, beta)
    Q = H;
    P = f;

    % L2 (Tikhonov) penalty terms
    if ~isempty(beta)
        for order = 1:length(beta)
            if beta(order) ~= 0
                % Get discrete gradient matrix of specified order
                L = GetGradMatrix(size(f, 1), order - 1);
                Q = Q + beta(order) * (L' * L);
            end
        end
    end

    % L1 (Lasso) penalty term
    if ~isempty(alpha)
        if isscalar(alpha)
            P = P + alpha(1);
        else
            error('Alpha must be a scalar.');
        end
    end
end

% --------------------- 
% @Description: Construct discrete gradient matrix L of given order
%   order=0: identity matrix I
%   order=1: first-difference matrix  (penalizes roughness)
%   order=2: second-difference matrix (penalizes curvature)
% --------------------- 
function L = GetGradMatrix(n, SmoothOrder)
    if SmoothOrder == 0
        L = spdiags(ones(n, 1), 0, n, n);
        return
    end

    % Build first-order difference matrix
    idy = [1:n, 1:n-1];
    idx = [1:n, 2:n];
    v = [ones(1, n), -ones(1, n-1)];
    L = sparse(idy, idx, v, n, n, 2*n - 1);

    % Raise to higher order if needed
    if n > SmoothOrder
        L = L^SmoothOrder;
        L(end - SmoothOrder + 1 : end, :) = [];  % Remove boundary rows
    end
end

%% ====================== Utility Functions ======================

% --------------------- 
% @Description: Interpolate matrix rows to a new dimension
%   Used to upsample wavelength axis for finer spectral resolution
% --------------------- 
function interpolatedMatrix = expand(originalMatrix, newRows)
    interpolatedMatrix = zeros(newRows, size(originalMatrix, 2));

    % Interpolate each column independently
    for col = 1:size(originalMatrix, 2)
        originalY = 1:size(originalMatrix, 1);
        newY = linspace(1, size(originalMatrix, 1), newRows);
        interpolatedMatrix(:, col) = interp1(originalY, originalMatrix(:, col), newY, 'pchip');
    end
end

% --------------------- 
% @Description: Visualize reconstructed spectrum vs ground truth
% --------------------- 
function ShowFigure(Solution, Wavelength, Data)
    hold on;
    % Ground truth spectrum (dashed)
    plot(Data.Wavelength, Data.Spectrum, '--', 'LineWidth', 1);
    % Reconstructed spectrum (solid)
    plot(Solution.Wavelength, Solution.Spectrum, '-', 'LineWidth', 1);
    xlabel('Wavelength (nm)'); ylabel('Intensity');
    set(gca, 'FontSize', 12, 'Ytick', -0.5:0.1:1);
    grid on; box on; set(gcf, 'Color', 'white');
    axis([Data.Wavelength(1), Data.Wavelength(end), -0.1, 1]);
    pbaspect([1 1 1]);

    % Mark ground-truth peak positions
    Peaks = Data.Peaks * (max(Solution.Wavelength) - min(Solution.Wavelength)) ...
            + min(Solution.Wavelength);
    for p = 1:length(Peaks)
        xline(Peaks(p), '--k', sprintf('x=%.1f nm', Peaks(p)), ...
            'LineWidth', 1, 'LabelHorizontalAlignment', 'left', 'FontSize', 10);
    end
end
