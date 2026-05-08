import numpy as np
from scipy import interpolate
from scipy.linalg import svd
from scipy.optimize import nnls, fmin_slsqp
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ================== Parameter settings ==================
NumWavelengths = 65
NumGaussianBasis = 65

MinLambda = 360
MaxLambda = 1000

FWHMset = 10

# ================== File names ==================
ResponseMatrixFile = 'ResponseMatrix.txt'

MeasuredSignalsFile = 'MeasuredSignals.txt'

ReferenceSpectraFile = 'refspectra.txt'

# ================== Load response matrix ==================
ResponseMatrix = np.loadtxt(ResponseMatrixFile)

if ResponseMatrix.shape[1] != NumWavelengths:
    raise ValueError(
        f"ResponseMatrix must have {NumWavelengths} columns "
        f"(wavelength samples). Got {ResponseMatrix.shape[1]}"
    )

# Normalize response matrix
ResponseMatrix = ResponseMatrix / np.max(
    ResponseMatrix,
    axis=1,
    keepdims=True
)

# ================== Wavelength grids ==================
VecOfLambdas = np.linspace(
    MinLambda,
    MaxLambda,
    NumWavelengths
) / MaxLambda

VecOfLambdasPlot = np.linspace(
    MinLambda,
    MaxLambda,
    NumWavelengths
) / MaxLambda

VecOfLambdasPlot_nm = VecOfLambdasPlot * MaxLambda

# ================== Interpolate response matrix ==================
NumRow = ResponseMatrix.shape[0]

ResponseMatrixInterp = np.zeros(
    (NumRow, NumWavelengths)
)

for r in range(NumRow):

    spline = interpolate.interp1d(
        VecOfLambdas * MaxLambda,
        ResponseMatrix[r, :],
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )

    ResponseMatrixInterp[r, :] = spline(
        VecOfLambdasPlot_nm
    )

# ================== Load measured signals ==================
MeasuredSignals = np.loadtxt(
    MeasuredSignalsFile
)

# Convert single spectrum to 2D array
if MeasuredSignals.ndim == 1:
    MeasuredSignals = MeasuredSignals[:, np.newaxis]

NumSpectra = MeasuredSignals.shape[1]

# ================== Load reference spectra ==================
ReferenceData = np.loadtxt(
    ReferenceSpectraFile
)

# First column = wavelength
ReferenceWavelength_nm = ReferenceData[:, 0]

# Remaining columns = spectra
ReferenceSpectra = ReferenceData[:, 1:]

# Convert single spectrum to 2D array
if ReferenceSpectra.ndim == 1:
    ReferenceSpectra = ReferenceSpectra[:, np.newaxis]

# ================== Reconstruction pipeline ==================
ReconstructedSpectrum = np.zeros(
    (NumWavelengths, NumSpectra)
)

# ================== GCV gamma finder ==================
def find_opt_gamma_gcv(c, A):

    U, s, Vt = svd(
        A,
        full_matrices=False
    )

    m = A.shape[0]

    s = np.maximum(s, 1e-12)

    def computeGCV(log_gamma):

        gamma = np.exp(log_gamma)

        fi = s**2 / (
            s**2 + gamma**2
        )

        Utc = U.T @ c

        residual = np.sum(
            ((1 - fi) * Utc)**2
        )

        denom = (
            m - np.sum(fi)
        )**2

        if denom < 1e-12:
            return 1e100

        return residual / denom

    try:

        log_gamma_opt = fmin_slsqp(
            lambda x: computeGCV(x[0]),
            x0=[-20.0],
            bounds=[(-50, 50)],
            disp=False
        )

        gamma = float(
            np.exp(log_gamma_opt[0])
        )

    except Exception:

        gamma = 1e-6

    return gamma

# ================== Gaussian basis ==================
GaussianCenter = np.linspace(
    MinLambda,
    MaxLambda,
    NumGaussianBasis
) / MaxLambda

GaussianSigma = (
    FWHMset
) / (
    MaxLambda *
    (2 * np.sqrt(2 * np.log(2)))
)

GaussianBasis = np.zeros(
    (
        NumWavelengths,
        NumGaussianBasis
    )
)

for j in range(NumGaussianBasis):

    GaussianBasis[:, j] = np.exp(
        -0.5 * (
            (
                VecOfLambdasPlot -
                GaussianCenter[j]
            ) / GaussianSigma
        ) ** 2
    )

# ================== Weight matrix ==================
WeightMatrix = (
    ResponseMatrixInterp @ GaussianBasis
)

# Normalize weight matrix
WeightMatrix = (
    WeightMatrix /
    np.max(WeightMatrix)
)

# ================== Regularization matrix ==================
L = np.eye(NumGaussianBasis)

# ================== Reconstruction loop ==================
for i in range(NumSpectra):

    y = MeasuredSignals[:, i].copy()

    # Normalize measured signals
    if np.max(y) > 0:
        y = y / np.max(y)

    # Find optimal gamma
    OptimalGamma = find_opt_gamma_gcv(
        y,
        WeightMatrix
    )

    sqrt_gamma = np.sqrt(
        OptimalGamma
    )

    # Construct augmented system
    AugWeightMatrix = np.vstack([
        WeightMatrix,
        sqrt_gamma * L
    ])

    AugMeasuredSignals = np.concatenate([
        y,
        np.zeros(L.shape[0])
    ])

    # Solve NNLS problem
    GaussianCoefficients, _ = nnls(
        AugWeightMatrix,
        AugMeasuredSignals
    )

    # Reconstruct spectrum
    recon = (
        GaussianBasis @
        GaussianCoefficients
    )

    if np.max(recon) > 0:
        recon = recon / np.max(recon)

    ReconstructedSpectrum[:, i] = recon

# ================== Plot spectra ==================
num_plot = min(6, NumSpectra)

plt.figure(
    figsize=(12, 4 * num_plot)
)

for i in range(num_plot):

    plt.subplot(
        num_plot,
        1,
        i + 1
    )

    # Plot reconstructed spectrum
    plt.plot(
        VecOfLambdasPlot_nm,
        ReconstructedSpectrum[:, i],
        'b-',
        linewidth=1.5,
        label='Reconstructed'
    )

    # Plot reference spectrum
    if i < ReferenceSpectra.shape[1]:

        ref = ReferenceSpectra[:, i]

        if np.max(ref) > 0:
            ref = ref / np.max(ref)

        plt.plot(
            ReferenceWavelength_nm,
            ref,
            'r--',
            linewidth=1.5,
            label='Reference'
        )

    plt.title(
        f'Spectrum {i + 1}'
    )

    plt.xlabel(
        'Wavelength (nm)'
    )

    plt.ylabel(
        'Normalized intensity'
    )

    plt.legend()

plt.tight_layout()

plt.show()

# ================== Save outputs ==================

# Use the same folder as the script
script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

# Generate timestamp
timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

# Generate output filename
output_filename = (
    f"reconstructed spectra_{timestamp}.txt"
)

output_path = os.path.join(
    script_folder,
    output_filename
)

# Save reconstructed spectra
ReconstructedToSave = np.column_stack((
    VecOfLambdasPlot_nm,
    ReconstructedSpectrum
))

np.savetxt(
    output_path,
    ReconstructedToSave,
    fmt='%.6f'
)

print(
    f'Reconstructed spectra saved to:\n'
    f'{output_path}'
)
