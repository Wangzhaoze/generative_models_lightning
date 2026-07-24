"""Radar ADC preprocessing with an explicit optional DoA branch."""

from __future__ import annotations

import numpy as np

from dataset_preprocessor.utils import radardsp


NOISE_THRESHOLD = 0.30


def _range_doppler_fft(radar_adc_data: np.ndarray, radar_config) -> np.ndarray:
    """ADC -> range FFT -> Doppler FFT and TDM velocity compensation."""
    ntx, nrx, _, num_samples = radar_adc_data.shape
    expected_antennas = (int(radar_config.numTxChan), int(radar_config.numRxChan))
    if (ntx, nrx) != expected_antennas:
        raise ValueError(
            f"ADC antenna shape {(ntx, nrx)} does not match {expected_antennas}"
        )

    windowed = radar_adc_data * np.blackman(num_samples).reshape(1, 1, 1, -1)
    spectrum = np.fft.fft(
        windowed,
        int(radar_config.range_fftsize),
        axis=-1,
    )
    spectrum = np.fft.fft(
        spectrum,
        int(radar_config.doppler_fftsize),
        axis=-2,
    )
    spectrum = np.fft.fftshift(spectrum, axes=-2)
    spectrum *= radardsp.velocity_compensation(
        ntx,
        int(radar_config.doppler_fftsize),
    )
    return spectrum


def _crop_range_bins(spectrum: np.ndarray, radar_config) -> np.ndarray:
    crop_low = int(spectrum.shape[-1] * float(radar_config.crop_low))
    crop_high = int(spectrum.shape[-1] * float(radar_config.crop_high))
    if crop_low:
        spectrum[..., :crop_low] = 0
    if crop_high:
        spectrum[..., -crop_high:] = 0
    return spectrum


def _collapse_doppler(
    spectrum: np.ndarray,
    radar_config,
    *,
    angular: bool,
) -> np.ndarray:
    """Collapse Doppler and return ``(range, axis_1, axis_2, I/V/mask)``."""
    if spectrum.shape[2] < 2:
        raise ValueError("At least two Doppler bins are required")

    if angular:
        elevation_bins, azimuth_bins, velocity_bins_count, range_bins = spectrum.shape
        _, velocity_bins, _, _ = radardsp._get_bins(
            range_bins,
            velocity_bins_count,
            azimuth_bins,
            elevation_bins,
            radar_config,
        )
    else:
        _, _, velocity_bins_count, range_bins = spectrum.shape
        _, velocity_bins, _, _ = radardsp._get_bins(
            range_bins,
            velocity_bins_count,
            0,
            0,
            radar_config,
        )

    power = np.abs(spectrum) ** 2
    max_indices = np.argmax(power, axis=2)
    velocity = np.transpose(velocity_bins[max_indices], (2, 1, 0))

    sorted_power = np.sort(power, axis=2)
    validity = (
        sorted_power[:, :, -1, :] * (1.0 - NOISE_THRESHOLD)
        > sorted_power[:, :, -2, :]
    )
    validity = np.transpose(validity, (2, 1, 0))

    integrated_power = np.sum(power, axis=2)
    noise = float(np.quantile(integrated_power, NOISE_THRESHOLD))
    intensity = 10.0 * np.log10(integrated_power / (noise + 1.0e-6) + 1.0)
    intensity = np.transpose(intensity, (2, 1, 0))

    result = np.stack((intensity, velocity, validity), axis=-1).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("Radar preprocessing produced NaN or Inf")
    return result


def RAEIVVmap(
    radar_adc_data: np.ndarray,
    radar_config,
    tx_array: np.ndarray,
    rx_array: np.ndarray,
    *,
    use_doa: bool = False,
    allow_aperture_truncation: bool = False,
) -> np.ndarray:
    """Create one radar cube with or without direction-of-arrival estimation.

    ``use_doa=False`` performs unpadded FFTs on the physical Rx and Tx axes and
    returns ``(range, rx, tx, 3)``.  It does not construct a virtual array.
    ``use_doa=True`` returns ``(range, azimuth, elevation, 3)``.
    The last axis is intensity, velocity, and Doppler-peak validity.
    """
    spectrum = _range_doppler_fft(radar_adc_data, radar_config)

    if use_doa:
        virtual_array = radardsp.virtual_array(spectrum, tx_array, rx_array)
        virtual_elevation, virtual_azimuth = virtual_array.shape[:2]
        angle_size = int(radar_config.ANGLE_fftsize)
        elevation_size = int(radar_config.ELEVATION_fftsize)
        if not allow_aperture_truncation and (
            angle_size < virtual_azimuth or elevation_size < virtual_elevation
        ):
            raise ValueError(
                "DoA FFT is smaller than the virtual aperture: "
                f"FFT={(angle_size, elevation_size)}, "
                f"aperture={(virtual_azimuth, virtual_elevation)}"
            )

        spectrum = np.fft.fft(virtual_array, angle_size, axis=1)
        spectrum = np.fft.fftshift(spectrum, axes=1)
        spectrum = np.fft.fft(spectrum, elevation_size, axis=0)
        spectrum = np.fft.fftshift(spectrum, axes=0)
        spectrum = _crop_range_bins(spectrum, radar_config)
        result = _collapse_doppler(spectrum, radar_config, angular=True)
        expected_shape = (
            int(radar_config.range_fftsize),
            angle_size,
            elevation_size,
            3,
        )
    else:
        # Keep the physical-array resolution: FFT length equals the number of
        # channels, so this branch performs no angular zero-padding.
        spectrum = np.fft.fft(
            spectrum,
            int(radar_config.numRxChan),
            axis=1,
        )
        spectrum = np.fft.fftshift(spectrum, axes=1)
        spectrum = np.fft.fft(
            spectrum,
            int(radar_config.numTxChan),
            axis=0,
        )
        spectrum = np.fft.fftshift(spectrum, axes=0)
        spectrum = _crop_range_bins(spectrum, radar_config)
        result = _collapse_doppler(spectrum, radar_config, angular=True)
        expected_shape = (
            int(radar_config.range_fftsize),
            int(radar_config.numRxChan),
            int(radar_config.numTxChan),
            3,
        )

    if result.shape != expected_shape:
        raise ValueError(
            f"Unexpected radar cube shape {result.shape}; expected {expected_shape}"
        )
    return result


__all__ = ["RAEIVVmap"]
