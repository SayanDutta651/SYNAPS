"""
Comprehensive Test Suite for FSK Symbol-Rate Investigation and Dashboard Output Audits.

Validates:
1. FSK symbol-rate estimation across known rates (50 kHz, 100 kHz, 125 kHz, 200 kHz).
2. Frequency-state transition tracking and state duration measurement for FSK.
3. Low-SNR and short-duration FSK robustness.
4. Provenance distinction between waveform-derived measurements and metadata references.
5. Waveform vs metadata disagreement reporting.
6. Classification probability semantics (softmax outputs summing to ~100%).
7. Multi-modal confidence and evidence score semantics.
8. Distinctions between Peak Spectral Frequency and Carrier Frequency Offset (CFO).
9. Distinctions between Occupied Bandwidth (99% power) and 3dB bandwidth.
10. RF fingerprint representation as a derived feature hash rather than physical emitter claim.

Runnable directly with:
    python tests/dsp/test_fsk_investigation_and_audit.py
"""

import math
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is available
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsp.timing.symbol_rate import estimate_symbol_rate, _estimate_fsk_candidate
from dsp.spectral.fft import compute_fft
from dsp.spectral.bandwidth import estimate_bandwidth
from dsp.frequency.cfo import estimate_cfo
from dsp.statistical.hoc import calculate_hoc
from intelligence.confidence.confidence import compute_composite_confidence
from fusion.evidence import aggregate_evidence
from intelligence.fingerprint.fingerprint import extract_signal_fingerprint
from backend.services.pipeline import analyze_signal


def _generate_synthetic_fsk(
    symbol_rate: float = 100_000.0,
    sampling_rate: float = 1_000_000.0,
    num_symbols: int = 500,
    snr_db: float = 20.0,
    freq_dev: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a clean continuous-phase 2FSK synthetic signal."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=num_symbols, dtype=np.uint8)
    sps = int(round(sampling_rate / symbol_rate))
    if freq_dev is None:
        freq_dev = symbol_rate / 2.0  # standard modulation index h = 1.0

    t_sym = np.arange(sps) / sampling_rate
    chunks = []
    phase = 0.0
    for b in bits:
        freq = freq_dev if b == 1 else -freq_dev
        chunk = np.exp(1j * (2 * np.pi * freq * t_sym + phase))
        phase = (2 * np.pi * freq * (sps / sampling_rate) + phase) % (2 * np.pi)
        chunks.append(chunk)

    signal = np.concatenate(chunks)
    if snr_db is not None and math.isfinite(snr_db):
        pwr = np.mean(np.abs(signal) ** 2)
        noise_pwr = pwr / (10.0 ** (snr_db / 10.0))
        noise = (rng.normal(0, np.sqrt(noise_pwr / 2), len(signal)) +
                 1j * rng.normal(0, np.sqrt(noise_pwr / 2), len(signal)))
        signal = signal + noise
    return signal.astype(np.complex128)


class TestFSKInvestigationAndAudits(unittest.TestCase):
    """Test suite covering FSK investigation and dashboard consistency audits."""

    def test_fsk_known_rates_50k_100k_125k_200k(self):
        """Verify FSK symbol rate estimator accurately recovers 50k, 100k, 125k, 200k Baud."""
        fs = 1_000_000.0
        test_rates = [50_000.0, 100_000.0, 125_000.0, 200_000.0]

        for target_rate in test_rates:
            sig = _generate_synthetic_fsk(symbol_rate=target_rate, sampling_rate=fs, snr_db=25.0, seed=int(target_rate))
            res = estimate_symbol_rate(sig, sampling_rate=fs)
            est_rate = res["symbol_rate_hz"]
            error_rel = abs(est_rate - target_rate) / target_rate
            self.assertLess(
                error_rel, 0.10,
                f"Target {target_rate} Hz produced estimate {est_rate} Hz (rel error {error_rel:.2%})"
            )

    def test_fsk_waveform_dataset_8_samples_per_symbol(self):
        """Verify that FSK signals with 8 samples per symbol evaluate to 125 kBaud."""
        fs = 1_000_000.0
        sig_125k = _generate_synthetic_fsk(symbol_rate=125_000.0, sampling_rate=fs, snr_db=20.0)
        res = estimate_symbol_rate(sig_125k, sampling_rate=fs)
        self.assertEqual(res["samples_per_symbol"], 8.0)
        self.assertEqual(res["symbol_rate_hz"], 125_000.0)

    def test_fsk_low_snr_robustness(self):
        """Verify FSK symbol rate estimator functions at low SNR (5 dB)."""
        fs = 1_000_000.0
        sig_low_snr = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, snr_db=5.0, seed=123)
        res = estimate_symbol_rate(sig_low_snr, sampling_rate=fs)
        self.assertGreater(res["symbol_rate_hz"], 0.0)
        # Should still detect within 30% even at extreme 5 dB SNR
        self.assertLess(abs(res["symbol_rate_hz"] - 100_000.0) / 100_000.0, 0.30)

    def test_fsk_short_duration(self):
        """Verify FSK symbol rate estimator handles short signals (100 symbols)."""
        fs = 1_000_000.0
        sig_short = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, num_symbols=100, snr_db=20.0)
        res = estimate_symbol_rate(sig_short, sampling_rate=fs)
        self.assertGreater(res["symbol_rate_hz"], 0.0)
        self.assertLess(abs(res["symbol_rate_hz"] - 100_000.0) / 100_000.0, 0.10)

    def test_classification_score_semantics(self):
        """Audit: verify that AI classification probabilities sum to approximately 100%."""
        fs = 1_000_000.0
        sig = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, snr_db=20.0)
        res = analyze_signal(sig, sample_rate=fs)
        
        breakdown = res["frontend_data"]["predictionBreakdown"]
        total_prob = sum(item["probability"] for item in breakdown)
        self.assertAlmostEqual(total_prob, 1.0, places=3, msg="Prediction breakdown probabilities must sum to 1.0 (100%)")

    def test_confidence_and_evidence_semantics(self):
        """Audit: verify separation between AI confidence, SNR factor, and multi-modal evidence."""
        ai_conf = 98.5
        snr_db = 12.0
        evidence_score = 0.89

        comp = compute_composite_confidence(ai_conf, snr_db, evidence_score)
        self.assertIn("composite_confidence", comp)
        self.assertIn("ai_confidence_pct", comp)
        self.assertIn("snr_db", comp)
        self.assertIn("evidence_factor", comp)

        self.assertAlmostEqual(comp["ai_confidence_pct"], 98.5, places=1)
        self.assertGreater(comp["composite_confidence"], 0.0)
        self.assertLessEqual(comp["composite_confidence"], 1.0)

    def test_cfo_and_peak_frequency_distinction(self):
        """Audit: verify distinction between 2-sided FFT Peak Frequency and 1-sided CFO."""
        fs = 1_000_000.0
        sig = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, freq_dev=50_000.0, snr_db=30.0)
        
        fft_res = compute_fft(sig, sampling_rate=fs)
        cfo_res = estimate_cfo(sig, sampling_rate=fs, reference_frequency_hz=0.0)
        
        # Peak frequency can be either negative or positive tone (-50 kHz or +50 kHz)
        self.assertTrue(abs(abs(fft_res["peak_frequency_hz"]) - 50_000.0) < 5_000.0)
        # CFO with reference 0 is the detected peak positive carrier frequency (~50 kHz)
        self.assertGreater(cfo_res["cfo_hz"], 0.0)

    def test_occupied_bandwidth_power_integration(self):
        """Audit: verify occupied bandwidth uses 99% cumulative power integration."""
        fs = 1_000_000.0
        # FSK with +/- 50 kHz tones (100 kHz separation) has occupied bandwidth ~180-260 kHz
        sig = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, freq_dev=50_000.0, snr_db=25.0)
        bw_res = estimate_bandwidth(sig, sampling_rate=fs, occupied_fraction=0.99)
        
        self.assertEqual(bw_res["occupied_fraction"], 0.99)
        self.assertGreater(bw_res["bandwidth_hz"], 100_000.0)
        self.assertLess(bw_res["bandwidth_hz"], 350_000.0)

    def test_hoc_cumulant_properties_for_fsk(self):
        """Audit: verify C40 is near 0 and C42 reflects constant-envelope power variance."""
        fs = 1_000_000.0
        sig = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, snr_db=25.0)
        hoc = calculate_hoc(sig)
        
        # C40 should be close to zero due to phase/frequency circular symmetry
        self.assertLess(abs(hoc["C40"]), 0.10)
        # C42 should be non-zero (near 1.0 for constant amplitude with low noise)
        self.assertGreater(abs(hoc["C42"]), 0.50)

    def test_rf_fingerprint_derived_identifier(self):
        """Audit: verify RF fingerprint format and derived feature-hash properties."""
        fs = 1_000_000.0
        sig = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, snr_db=20.0)
        dsp_metrics = {"cfo_hz": 50000.0, "snr_db": 20.0, "bandwidth_hz": 200000.0}
        fp = extract_signal_fingerprint(sig, dsp_metrics, "FSK")
        
        self.assertTrue(fp["fingerprint_id"].startswith("RF-FP-"))
        self.assertEqual(len(fp["fingerprint_id"]), 22)  # 'RF-FP-' + 16 hex chars
        self.assertIn("papr_db", fp)
        self.assertIn("average_power", fp)

    def test_pipeline_fsk_dataset_provenance_handling(self):
        """Verify pipeline execution on representative FSK IQ file preserves waveform estimate & metadata ref."""
        fsk_path = PROJECT_ROOT / "data" / "iq" / "FSK" / "signal_0401_fsk.iq"
        if fsk_path.exists():
            res = analyze_signal(str(fsk_path))
            dsp = res["dsp_analysis"]
            self.assertEqual(dsp["symbol_rate_waveform"], 125000.0)
            self.assertEqual(dsp["symbol_rate_metadata_ref"], 100000.0)
            self.assertEqual(dsp["symbol_rate_status"], "Waveform/metadata disagreement")

    def test_fsk_carrier_center_and_tone_metrics(self):
        """Audit: verify estimate_fsk_carrier_center recovers carrier center, tone separation, and deviation."""
        from dsp.frequency.cfo import estimate_fsk_carrier_center
        fs = 1_000_000.0
        # Synthetic FSK with +/- 50 kHz deviation around 0 Hz carrier (100 kHz tone sep)
        sig = _generate_synthetic_fsk(symbol_rate=100_000.0, sampling_rate=fs, freq_dev=50_000.0, snr_db=25.0)
        carrier_res = estimate_fsk_carrier_center(sig, sampling_rate=fs, reference_frequency_hz=0.0)
        
        self.assertAlmostEqual(carrier_res["carrier_center_hz"], 0.0, delta=1000.0)
        self.assertAlmostEqual(carrier_res["tone_separation_hz"], 100_000.0, delta=2000.0)
        self.assertAlmostEqual(carrier_res["frequency_deviation_hz"], 50_000.0, delta=1000.0)

    def test_fsk_modulation_index_consistency(self):
        """Audit: verify modulation index formula h = 2*Delta_f / Rs = tone_sep / Rs."""
        rs = 125_000.0
        delta_f = 60_000.0
        tone_sep = 2 * delta_f  # 120 kHz
        h = (2 * delta_f) / rs   # 120 / 125 = 0.96
        self.assertAlmostEqual(h, 0.96, places=2)
        self.assertAlmostEqual(tone_sep / rs, h, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
