function pPars = Custom_3T_Phantom_PeakBounds(tube_type)
% Custom_3T_Phantom_PeakBounds: Customized peak fitting bounds for your
% 24-tube 3T phantom
%
% INPUT:
%   tube_type - String specifying tube content:
%               'iopamidol', 'creatine', 'taurine', 'PLL', or 'PBS'
%
% OUTPUT:
%   pPars - Struct with start points, lower bounds, upper bounds for
%           Lorentzian peak fitting
%
% USAGE:
%   pPars = Custom_3T_Phantom_PeakBounds('iopamidol');
%   pPars = Custom_3T_Phantom_PeakBounds('creatine');
%
% Author: Custom configuration for 3T phantom analysis
% Date: 2025-12-02

% Start with default bounds
pPars = zspecSetLPeakBounds();

% Water peak is always present (fixed at 0 ppm)
pPars.water.st = [0.9, 1.4, 0, 0];
pPars.water.lb = [0.5, 0.5, -0.1, 0];
pPars.water.ub = [1.0, 3.0, 0.1, 0];

switch lower(tube_type)
    case 'iopamidol'
        % Iopamidol has TWO main peaks at 4.3 and 5.5 ppm
        % We'll use 'amine' for the 4.3 ppm peak (strongest)
        % and 'amide' for the 5.5 ppm peak

        % Peak 1: 4.3 ppm (strongest)
        pPars.amine.st = [0.15, 0.8, 4.3, 0];    % Amplitude, FWHM, offset, phase
        pPars.amine.lb = [0.01, 0.3, 4.0, 0];
        pPars.amine.ub = [0.5, 2.0, 4.6, 0];

        % Peak 2: 5.5 ppm (secondary)
        pPars.amide.st = [0.10, 0.7, 5.5, 0];
        pPars.amide.lb = [0.01, 0.3, 5.2, 0];
        pPars.amide.ub = [0.4, 2.0, 5.8, 0];

    case 'creatine'
        % Creatine: single peak at 1.9 ppm
        pPars.amine.st = [0.12, 0.6, 1.9, 0];
        pPars.amine.lb = [0.01, 0.2, 1.7, 0];
        pPars.amine.ub = [0.5, 1.5, 2.1, 0];

    case 'taurine'
        % Taurine: peak at ~2.8 ppm
        pPars.amine.st = [0.10, 0.7, 2.8, 0];
        pPars.amine.lb = [0.01, 0.3, 2.6, 0];
        pPars.amine.ub = [0.4, 1.5, 3.0, 0];

    case 'pll'
        % Poly-L-lysine: amide peak at ~3.5 ppm and amine at ~2.7 ppm
        % Use 'amide' for 3.5 ppm
        pPars.amide.st = [0.08, 0.8, 3.5, 0];
        pPars.amide.lb = [0.01, 0.3, 3.3, 0];
        pPars.amide.ub = [0.4, 2.0, 3.7, 0];

        % Use 'amine' for 2.7 ppm
        pPars.amine.st = [0.06, 0.7, 2.7, 0];
        pPars.amine.lb = [0.01, 0.3, 2.5, 0];
        pPars.amine.ub = [0.3, 1.5, 2.9, 0];

    case 'pbs'
        % PBS blank: only water peak (minimal CEST effect)
        % Still fit for a small amine peak to capture any residual signal
        pPars.amine.st = [0.02, 0.5, 2.0, 0];
        pPars.amine.lb = [0.0, 0.2, 1.0, 0];
        pPars.amine.ub = [0.1, 1.5, 4.0, 0];

    otherwise
        error('Unknown tube type: %s. Use: iopamidol, creatine, taurine, PLL, or PBS', tube_type);
end

end
