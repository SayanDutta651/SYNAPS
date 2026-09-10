"""Controlled symbol-rate tests for waveform-derived timing estimates."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsp.timing.symbol_rate import estimate_symbol_rate


SAMPLE_RATE = 1_000_000.0


def _make_signal(modulation: str, symbol_rate: float, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples_per_symbol = int(round(SAMPLE_RATE / symbol_rate))
    number_of_symbols = 400

    if modulation == "QPSK":
        bits = rng.integers(0, 2, size=number_of_symbols * 2)
        symbols = (
            np.where(bits[0::2] == 0, 1.0, -1.0)
            + 1j * np.where(bits[1::2] == 0, 1.0, -1.0)
        ) / np.sqrt(2.0)
        clean = np.repeat(symbols, samples_per_symbol)
    elif modulation == "QAM16":
        bits = rng.integers(0, 2, size=number_of_symbols * 4)
        levels = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
        symbols = np.array(
            [
                (
                    levels[tuple(bits[index:index + 2])]
                    + 1j * levels[tuple(bits[index + 2:index + 4])]
                ) / np.sqrt(10.0)
                for index in range(0, len(bits), 4)
            ]
        )
        clean = np.repeat(symbols, samples_per_symbol)
    elif modulation == "FSK":
        bits = rng.integers(0, 2, size=number_of_symbols)
        time = np.arange(samples_per_symbol) / SAMPLE_RATE
        phase = 0.0
        chunks = []
        for bit in bits:
            frequency = symbol_rate / 2.0
            frequency = frequency if bit else -frequency
            chunk = np.exp(1j * (2.0 * np.pi * frequency * time + phase))
            phase = np.angle(chunk[-1])
            chunks.append(chunk)
        clean = np.concatenate(chunks)
    else:
        raise ValueError(modulation)

    power = np.mean(np.abs(clean) ** 2)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=clean.size) + 1j * rng.normal(size=clean.size)
    )
    return clean + noise


class TestControlledSymbolRate(unittest.TestCase):
    def test_known_rates_at_20_db(self):
        for modulation in ("QPSK", "QAM16", "FSK"):
            for symbol_rate in (50_000.0, 100_000.0, 200_000.0):
                signal = _make_signal(modulation, symbol_rate, 20.0, 42)
                result = estimate_symbol_rate(
                    signal,
                    SAMPLE_RATE,
                    minimum_symbol_rate_hz=20_000.0,
                    maximum_symbol_rate_hz=300_000.0,
                )
                error = abs(result["symbol_rate_hz"] - symbol_rate) / symbol_rate
                tolerance = 0.10 if modulation == "FSK" else 0.05
                self.assertLessEqual(
                    error,
                    tolerance,
                    msg=f"{modulation} {symbol_rate}: {result}",
                )
                self.assertEqual(result["symbol_rate_status"], "detected")
                self.assertTrue(result["candidate_rates"])

    def test_known_rates_across_snr(self):
        for modulation in ("QPSK", "QAM16", "FSK"):
            for snr_db in (15.0, 10.0, 5.0):
                signal = _make_signal(modulation, 100_000.0, snr_db, 100 + int(snr_db))
                result = estimate_symbol_rate(
                    signal,
                    SAMPLE_RATE,
                    minimum_symbol_rate_hz=20_000.0,
                    maximum_symbol_rate_hz=300_000.0,
                )
                self.assertIn(result["symbol_rate_status"], ("detected", "low_confidence"))
                self.assertGreater(result["symbol_rate_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
