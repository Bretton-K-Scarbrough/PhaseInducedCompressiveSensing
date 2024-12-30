import numpy as np
import time
from typing import Tuple, List


def focus_phase(
    lam: float, X: np.ndarray, Y: np.ndarray, f: float, x0: float, y0: float, z0: float
) -> np.ndarray:
    """
    Calculates the phase profile required to focus light to a given point.

    Parameters:
    lam (float): Wavelength of light.
    X (np.ndarray): X-coordinates of the pixel grid.
    Y (np.ndarray): Y-coordinates of the pixel grid.
    f (float): Focal length of system's equivalent focal length.
    x0 (float): Target x-position for focusing.
    y0 (float): Target y-position for focusing.
    z0 (float): Target z-position for focusing.

    Returns:
    np.ndarray: Phase profile required to focus on the target.
    """
    k = 2 * np.pi / lam
    trap_phase = (k / f) * (x0 * X + y0 * Y) + (np.pi * z0) / (lam * f**2) * (
        X**2 + Y**2
    )
    return trap_phase


def rs(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Runs the Random Superposition (RS) algorithm to calculate phasemask for optical traps

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A 1D array of the phase correction terms for each trap
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """
    t = time.perf_counter()

    pists = np.random.random(x0.shape[0]) * 2 * np.pi
    total_pixels = int(X.shape[0] * X.shape[1])

    # Generate individual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Calculate the superposition of each trap
    slm_total_field = np.sum(
        1.0 / total_pixels * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    t = time.perf_counter() - t
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)

    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))

    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    # put into SLM shape
    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, pists, [efficiency, uniformity, variance, t]


def gs(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    iters: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Runs the Weighted Gerchberg-Saxton (CSGS) algorithm to optimize phase settings
    for holographic optical trapping.

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A 1D array of the phase correction terms for each trap
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """
    t = time.perf_counter()

    # Load pists and create coordinate system
    pists = np.random.random(x0.shape[0]) * 2 * np.pi
    total_pixels = np.asarray(X.shape[0] * X.shape[1])

    # Generate invididual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Main GS loop
    for i in range(iters):
        slm_total_field = np.sum(
            1.0 / total_pixels * np.exp(1j * (individual_trap_phases + pists[:, None])),
            axis=0,
        )
        slm_total_phase = np.angle(slm_total_field)

        spot_fields = np.sum(
            1.0
            / total_pixels
            * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
            axis=1,
        )
        pists = np.angle(spot_fields)
        print(f"gs iter {i+1} out of {iters}")
    # Evaluate preformance
    t = time.perf_counter() - t
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)

    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))

    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, pists, [efficiency, uniformity, variance, t]


def wgs(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    iters: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Runs the Weighted Gerchberg-Saxton (WGS) algorithm to optimize phase settings
    for holographic optical trapping.

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).
    sub (float): Fraction of pixels to be used for subsampling in each iteration.

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A 1D array of the phase correction terms for each trap
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """
    t = time.perf_counter()

    # Initialize pists, weights, and create coordinate system
    pists = np.random.random(x0.shape[0]) * 2 * np.pi
    weights = np.ones(x0.shape[0]) / float(x0.shape[0])
    total_pixels = np.asarray(X.shape[0] * X.shape[1])

    # Generate invididual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Main GS loop
    for i in range(iters):
        slm_total_field = np.sum(
            (weights[:, None] / total_pixels)
            * np.exp(1j * (individual_trap_phases + pists[:, None])),
            axis=0,
        )
        slm_total_phase = np.angle(slm_total_field)

        spot_fields = np.sum(
            1.0
            / total_pixels
            * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
            axis=1,
        )
        pists = np.angle(spot_fields)
        ints = np.abs(spot_fields) ** 2

        weights = weights * (np.mean(np.sqrt(ints)) / np.sqrt(ints))
        weights = weights / np.sum(weights)

        print(f"wgs iter {i+1} out of {iters}")
    t = time.perf_counter() - t

    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)

    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))

    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, pists, [efficiency, uniformity, variance, t]


def csgs(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    sub: float,
    iters: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Runs the Compressive Sensing Gerchberg-Saxton (CSGS) algorithm to optimize phase settings
    for holographic optical trapping.

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).
    sub (float): Fraction of pixels to be used for subsampling in each iteration.
    iters (int): Number of iterations of the Gerchberg-Saxton algorithm to run.

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A 1D array of the phase correction terms for each trap
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """
    t = time.perf_counter()
    # Load pists and create coordinate system

    pists = np.random.random(x0.shape[0]) * 2 * np.pi
    total_pixels = int(X.shape[0] * X.shape[1])
    coordslist = np.asarray(range(total_pixels - 1))
    np.random.shuffle(coordslist)

    # Generate individual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Main GS loop
    for i in range(iters - 1):
        coordslist = np.roll(coordslist, int(coordslist.shape[0] * sub))
        coordlist_sparse = coordslist[: int(coordslist.shape[0] * sub)]

        slm_total_field = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j * (individual_trap_phases[:, coordlist_sparse] + pists[:, None])
            ),
            axis=0,
        )
        slm_total_phase = np.angle(slm_total_field)

        spot_fields = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j
                * (
                    slm_total_phase[None, :]
                    - individual_trap_phases[:, coordlist_sparse]
                )
            ),
            axis=1,
        )
        pists = np.angle(spot_fields)
        print(f"pics iter {i+1} out of {iters}")
    slm_total_field = np.sum(
        1.0 / total_pixels * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    t = time.perf_counter() - t
    print(f"csgs iter {iters} out of {iters}")
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)

    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))

    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, pists, [efficiency, uniformity, variance, t]


def wcsgs(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    sub: float,
    iters: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Runs the Weighted Compressive Sensing Gerchberg-Saxton (CSGS) algorithm to optimize phase settings
    for holographic optical trapping.

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).
    sub (float): Fraction of pixels to be used for subsampling in each iteration.
    iters (int): Number of iterations of the Gerchberg-Saxton algorithm to run.

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A 1D array of the phase correction terms for each trap
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """
    t = time.perf_counter()

    # Load pists and create coordinate system
    pists = np.random.random(x0.shape[0]) * 2 * np.pi
    total_pixels = int(X.shape[0] * X.shape[1])
    coordslist = np.asarray(range(total_pixels - 1))
    np.random.shuffle(coordslist)

    # Generate individual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Main GS loop
    for i in range(iters):
        coordslist = np.roll(coordslist, int(coordslist.shape[0] * sub))
        coordlist_sparse = coordslist[: int(coordslist.shape[0] * sub)]

        slm_total_field = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j * (individual_trap_phases[:, coordlist_sparse] + pists[:, None])
            ),
            axis=0,
        )
        slm_total_phase = np.angle(slm_total_field)

        spot_fields = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j
                * (
                    slm_total_phase[None, :]
                    - individual_trap_phases[:, coordlist_sparse]
                )
            ),
            axis=1,
        )
        pists = np.angle(spot_fields)
        print(f"wcsgs iter {i+1} out of {iters}")
    slm_total_field = np.sum(
        1.0 / total_pixels * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    # Calculate spot fields and then intensities for weight calculation
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    # Calculate individual trap weights
    weights = (
        np.ones(x0.shape[0])
        / float(x0.shape[0])
        * (np.mean(np.sqrt(ints)) / np.sqrt(ints))
    )
    weights = weights / np.sum(weights)

    # Calculate total field with applied weights
    slm_total_field = np.sum(
        weights[:, None]
        / (total_pixels)
        * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    # Calculate Preformance
    t = time.perf_counter() - t
    print(f"wcsgs iter {iters} out of {iters}")
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)
    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))
    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, pists, [efficiency, uniformity, variance, t]


def pics(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    sub: float,
    iters: int,
    phi: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """
    Runs the Phase Induced Compressive Sensing (PICS) Gerchberg-Saxton algorithm to optimize phase settings
    for holographic optical trapping.

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).
    sub (float): Fraction of pixels to be used for subsampling in each iteration.
    iters (int): Number of iterations of the Gerchberg-Saxton algorithm to run.
    phi (np.ndarray): Previously calculated phase pattern

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """

    t = time.perf_counter()
    # Load pists and create coordinate system

    pists = phi
    total_pixels = int(X.shape[0] * X.shape[1])
    coordslist = np.asarray(range(total_pixels - 1))
    np.random.shuffle(coordslist)

    # Generate individual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Main GS loop
    for i in range(iters - 1):
        # Subsampling the pixel coordinates
        coordslist = np.roll(coordslist, int(coordslist.shape[0] * sub))
        coordlist_sparse = coordslist[: int(coordslist.shape[0] * sub)]

        slm_total_field = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j * (individual_trap_phases[:, coordlist_sparse] + pists[:, None])
            ),
            axis=0,
        )
        slm_total_phase = np.angle(slm_total_field)

        spot_fields = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j
                * (
                    slm_total_phase[None, :]
                    - individual_trap_phases[:, coordlist_sparse]
                )
            ),
            axis=1,
        )
        pists = np.angle(spot_fields)
        print(f"pics iter {i+1} out of {iters}")
    slm_total_field = np.sum(
        1.0 / total_pixels * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    t = time.perf_counter() - t
    print(f"pics iter {iters} out of {iters}")
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)

    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))

    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, [efficiency, uniformity, variance, t]


def piwcs(
    lam: float,
    X: np.ndarray,
    Y: np.ndarray,
    f: float,
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    sub: float,
    iters: int,
    phi: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """
    Runs the Phase Induced Weighted Compressive Sensing (PIWCS) Gerchberg-Saxton algorithm to optimize phase settings
    for holographic optical trapping.

    Parameters:
    lam (float): Wavelength of the laser light (in meters).
    X (np.ndarray): 2D array of X-coordinates for the pixel grid of the spatial light modulator (SLM).
    Y (np.ndarray): 2D array of Y-coordinates for the pixel grid of the SLM.
    f (float): Focal length of the optical system (in meters).
    x0 (np.ndarray): 1D array of target x-positions for the traps (in meters).
    y0 (np.ndarray): 1D array of target y-positions for the traps (in meters).
    z0 (np.ndarray): 1D array of target z-positions for the traps (in meters).
    sub (float): Fraction of pixels to be used for subsampling in each iteration.
    iters (int): Number of iterations of the Gerchberg-Saxton algorithm to run.
    phi (np.ndarray): Previously calculated phase pattern

    Returns:
    Tuple[np.ndarray, List[float]]:
        - A 2D array representing the optimized phase pattern on the SLM.
        - A list containing the following performance metrics:
            - Efficiency (float): The efficiency of the beam energy reaching the target spots.
            - Uniformity (float): Measure of how uniformly the energy is distributed across the traps.
            - Variance (float): Normalized variance of the intensities across the traps.
            - Runtime (float): Time taken to run the algorithm.
    """

    t = time.perf_counter()
    # Load pists and create coordinate system

    pists = phi
    total_pixels = int(X.shape[0] * X.shape[1])
    coordslist = np.asarray(range(total_pixels - 1))
    np.random.shuffle(coordslist)

    # Generate individual trap fields
    individual_trap_phases = np.zeros((x0.shape[0], total_pixels))
    for i in range(x0.shape[0]):
        individual_trap_phases[i, :] = focus_phase(
            lam, X, Y, f, x0[i], y0[i], z0[i]
        ).flatten()

    # Main GS loop
    for i in range(iters - 1):
        # Subsampling the pixel coordinates
        coordslist = np.roll(coordslist, int(coordslist.shape[0] * sub))
        coordlist_sparse = coordslist[: int(coordslist.shape[0] * sub)]

        slm_total_field = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j * (individual_trap_phases[:, coordlist_sparse] + pists[:, None])
            ),
            axis=0,
        )
        slm_total_phase = np.angle(slm_total_field)

        spot_fields = np.sum(
            1.0
            / total_pixels
            * np.exp(
                1j
                * (
                    slm_total_phase[None, :]
                    - individual_trap_phases[:, coordlist_sparse]
                )
            ),
            axis=1,
        )
        pists = np.angle(spot_fields)
        print(f"pics iter {i+1} out of {iters}")
    slm_total_field = np.sum(
        1.0 / total_pixels * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    # Calculate spot fields and tehn intensitites for weight calculation
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    # Calculate individual trap weights
    weights = (
        np.ones(x0.shape[0])
        / float(x0.shape[0])
        * (np.mean(np.sqrt(ints)) / np.sqrt(ints))
    )
    weights = weights / np.sum(weights)

    # Calculate total field with applied weights
    slm_total_field = np.sum(
        weights[:, None]
        / (total_pixels)
        * np.exp(1j * (individual_trap_phases + pists[:, None])),
        axis=0,
    )
    slm_total_phase = np.angle(slm_total_field)

    t = time.perf_counter() - t
    print(f"piwcs iter {iters} out of {iters}")
    # Calculate Preformance
    spot_fields = np.sum(
        1.0
        / total_pixels
        * np.exp(1j * (slm_total_phase[None, :] - individual_trap_phases[:])),
        axis=1,
    )
    ints = np.abs(spot_fields) ** 2

    efficiency = np.sum(ints)
    uniformity = 1 - (np.max(ints) - np.min(ints)) / (np.max(ints) + np.min(ints))
    variance = np.sqrt(np.var(ints)) / np.mean(ints)

    slm_total_phase = np.reshape(slm_total_phase, (1080, 1920))
    return slm_total_phase, [efficiency, uniformity, variance, t]
