use num_complex::Complex;
use rustfft::FftPlanner;

// Monotone Piecewise Cubic Hermite Interpolator (PCHIP)
pub struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>, // derivatives
}

impl Pchip {
    pub fn new_unsorted(x: Vec<f64>, y: Vec<f64>) -> Result<Self, &'static str> {
        if x.len() != y.len() || x.len() < 2 {
            return Err("Invalid input sizes");
        }
        // sort by x ascending, drop exact duplicates (keep last)
        let mut xy: Vec<(f64, f64)> = x.into_iter().zip(y.into_iter()).collect();
        xy.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut xs = Vec::with_capacity(xy.len());
        let mut ys = Vec::with_capacity(xy.len());
        for (i, (xf, yf)) in xy.into_iter().enumerate() {
            if i == 0 || xf > xs[xs.len() - 1] {
                xs.push(xf);
                ys.push(yf);
            } else {
                // overwrite last y if same x
                ys[xs.len() - 1] = yf;
            }
        }
        if xs.len() < 2 {
            return Err("Not enough unique x points");
        }

        let n = xs.len();
        let mut h = vec![0.0; n - 1];
        let mut delta = vec![0.0; n - 1];
        for i in 0..n - 1 {
            h[i] = xs[i + 1] - xs[i];
            if h[i] <= 0.0 {
                return Err("x must be strictly increasing");
            }
            delta[i] = (ys[i + 1] - ys[i]) / h[i];
        }

        // Fritsch-Carlson slopes (interior points)
        let mut d = vec![0.0; n];

        for i in 1..n - 1 {
            if delta[i - 1] * delta[i] <= 0.0 {
                d[i] = 0.0;
            } else {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }

        // Edge cases: use one-sided three-point estimate
        if n >= 2 {
            let h0 = h[0];
            let h1 = if n > 2 { h[1] } else { h[0] };
            let m0 = delta[0];
            let m1 = if n > 2 { delta[1] } else { delta[0] };

            let mut d0 = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);

            if d0.signum() != m0.signum() {
                d0 = 0.0;
            } else if m0.signum() != m1.signum() && d0.abs() > 3.0 * m0.abs() {
                d0 = 3.0 * m0;
            }

            d[0] = d0;
        }

        if n >= 2 {
            let h0 = h[n - 2];
            let h1 = if n > 2 { h[n - 3] } else { h[n - 2] };
            let m0 = delta[n - 2];
            let m1 = if n > 2 { delta[n - 3] } else { delta[n - 2] };

            let mut d_last = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);

            if d_last.signum() != m0.signum() {
                d_last = 0.0;
            } else if m0.signum() != m1.signum() && d_last.abs() > 3.0 * m0.abs() {
                d_last = 3.0 * m0;
            }

            d[n - 1] = d_last;
        }

        Ok(Self { x: xs, y: ys, d })
    }

    pub fn interpolate_with_extrapolation(&self, xq: f64, use_linear_extrap: bool) -> f64 {
        let n = self.x.len();

        if xq <= self.x[0] {
            if use_linear_extrap {
                return self.y[0] + self.d[0] * (xq - self.x[0]);
            } else {
                return self.y[0];
            }
        }

        if xq >= self.x[n - 1] {
            if use_linear_extrap {
                return self.y[n - 1] + self.d[n - 1] * (xq - self.x[n - 1]);
            } else {
                return self.y[n - 1];
            }
        }

        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= xq {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let h = self.x[lo + 1] - self.x[lo];
        let t = (xq - self.x[lo]) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * self.y[lo] + h10 * h * self.d[lo] + h01 * self.y[lo + 1] + h11 * h * self.d[lo + 1]
    }

    /// Interpolate with separate extrapolation control for DC and HF
    /// For HF roll-off mode: uses exponential decay instead of linear to avoid oscillations
    pub fn interpolate_with_extrap_modes(
        &self,
        xq: f64,
        use_dc_linear: bool,
        use_hf_linear: bool,
    ) -> f64 {
        let n = self.x.len();

        if xq <= self.x[0] {
            if use_dc_linear {
                return self.y[0] + self.d[0] * (xq - self.x[0]);
            } else {
                return self.y[0];
            }
        }

        if xq >= self.x[n - 1] {
            if use_hf_linear {
                // Exponential decay for smooth roll-off (avoids oscillations)
                // Uses the last two points to estimate a decay rate
                let x_last = self.x[n - 1];
                let y_last = self.y[n - 1];
                let dx = xq - x_last;

                // Compute decay constant from the derivative at the last point
                // For a smooth roll-off, we want the magnitude to decay exponentially
                // If derivative is negative (roll-off), use exponential decay
                // Otherwise, clamp to zero to avoid growing magnitude
                if self.d[n - 1] < 0.0 && y_last > 0.0 {
                    // Exponential decay: y(x) = y_last * exp(lambda * dx)
                    // where lambda = d[n-1] / y_last (ensures smooth transition)
                    let lambda = self.d[n - 1] / y_last;
                    let result = y_last * (lambda * dx).exp();
                    return result.max(1e-20); // Floor at very small positive value
                } else {
                    // If already small or derivative is positive, just return last value
                    return y_last.max(1e-20);
                }
            } else {
                return self.y[n - 1];
            }
        }

        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= xq {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let h = self.x[lo + 1] - self.x[lo];
        let t = (xq - self.x[lo]) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * self.y[lo] + h10 * h * self.d[lo] + h01 * self.y[lo + 1] + h11 * h * self.d[lo + 1]
    }
}

/// Parse FRD text content into (frequency_Hz, magnitude_dB, phase_deg) tuples.
pub fn parse_frd(content: &str) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') || t.starts_with('*') {
            continue;
        }
        let cols: Vec<&str> = t
            .split(|c: char| c.is_whitespace() || c == ',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if cols.len() < 3 {
            continue;
        }
        if let (Ok(f), Ok(m_db), Ok(p_deg)) = (
            cols[0].parse::<f64>(),
            cols[1].parse::<f64>(),
            cols[2].parse::<f64>(),
        ) {
            if f.is_finite() && f > 0.0 && m_db.is_finite() && p_deg.is_finite() {
                out.push((f, m_db, p_deg));
            }
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Unwrap phase to remove 2π discontinuities.
pub fn unwrap_phase_radians(phases: &[f64]) -> Vec<f64> {
    if phases.is_empty() {
        return vec![];
    }

    let mut unwrapped = vec![phases[0]];
    let mut cumulative_offset = 0.0;

    for i in 1..phases.len() {
        let diff = phases[i] - phases[i - 1];

        if diff > std::f64::consts::PI {
            cumulative_offset -= 2.0 * std::f64::consts::PI;
        } else if diff < -std::f64::consts::PI {
            cumulative_offset += 2.0 * std::f64::consts::PI;
        }

        unwrapped.push(phases[i] + cumulative_offset);
    }

    unwrapped
}

/// Resample a spectrum to reduce the number of points for logarithmic display
///
/// Progressive downsampling strategy optimized for logarithmic X-axis:
/// - Below 500 Hz: Keep all points (critical low frequency region)
/// - 500 Hz - 1000 Hz: Keep 1 out of 2 points
/// - 1000 Hz - 2000 Hz: Keep 1 out of 3 points
/// - 2000 Hz - 5000 Hz: Keep 1 out of 4 points
/// - 5000 Hz - 10000 Hz: Keep 1 out of 6 points
/// - 10000 Hz - 15000 Hz: Keep 1 out of 7 points
/// - Above 15000 Hz: Keep 1 out of 8 points
pub fn resample_plot_data(spectrum: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if spectrum.is_empty() {
        return Vec::new();
    }

    let mut resampled = Vec::with_capacity(spectrum.len() / 4);

    // Track skip counters for each frequency band
    let mut counter_500 = 0u32;
    let mut counter_1k = 0u32;
    let mut counter_2k = 0u32;
    let mut counter_5k = 0u32;
    let mut counter_10k = 0u32;
    let mut counter_15k = 0u32;

    for &(freq, mag) in spectrum.iter() {
        let should_keep = if freq > 15000.0 {
            counter_15k += 1;
            counter_15k % 8 == 0
        } else if freq > 10000.0 {
            counter_10k += 1;
            counter_10k % 7 == 0
        } else if freq > 5000.0 {
            counter_5k += 1;
            counter_5k % 6 == 0
        } else if freq > 2000.0 {
            counter_2k += 1;
            counter_2k % 4 == 0
        } else if freq > 1000.0 {
            counter_1k += 1;
            counter_1k % 3 == 0
        } else if freq > 500.0 {
            counter_500 += 1;
            counter_500 % 2 == 0
        } else {
            true
        };

        if should_keep {
            resampled.push((freq, mag));
        }
    }

    resampled
}

/// Reconstruct impulse response from FRD data with adjustable delay
pub fn impulse_from_frd(
    frd: &[(f64, f64, f64)],
    fs: f64,
    delay_ms: f64,
) -> (Vec<f64>, Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let nyq = fs / 2.0;

    let mut data: Vec<(f64, f64, f64)> = frd
        .iter()
        .cloned()
        .filter(|(f, _, _)| *f <= nyq && f.is_finite())
        .collect();

    if data.len() < 2 {
        return (vec![0.0; fs.round() as usize], vec![], vec![]);
    }

    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let freqs: Vec<f64> = data.iter().map(|(f, _, _)| *f).collect();
    let mags_db: Vec<f64> = data.iter().map(|(_, m_db, _)| *m_db).collect();
    let phases_deg: Vec<f64> = data.iter().map(|(_, _, p_deg)| *p_deg).collect();

    let mags_lin: Vec<f64> = mags_db
        .iter()
        .map(|m_db| 10f64.powf(*m_db / 20.0))
        .collect();

    let phases_rad_wrapped: Vec<f64> = phases_deg.iter().map(|p_deg| p_deg.to_radians()).collect();
    let phases_rad: Vec<f64> = unwrap_phase_radians(&phases_rad_wrapped);

    let delay_compensation = delay_ms / 1000.0;

    let mag_interp = Pchip::new_unsorted(freqs.clone(), mags_lin)
        .expect("Failed to create magnitude interpolator");
    let phase_interp = Pchip::new_unsorted(freqs.clone(), phases_rad)
        .expect("Failed to create phase interpolator");

    let fft_size = 4 * (fs as usize).next_power_of_two();
    let half = fft_size / 2;

    let mut spectrum: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];

    // Store interpolated data for display
    let mut interp_mag_db = Vec::new();
    let mut interp_phase_deg = Vec::new();

    for k in 0..=half {
        let f_bin = k as f64 * fs / fft_size as f64;

        // Interpolate magnitude and phase (constant extrapolation)
        let mut mag = mag_interp.interpolate_with_extrap_modes(f_bin, false, false);
        let phase_interp_raw = phase_interp.interpolate_with_extrapolation(f_bin, false);
        let mut phase = phase_interp_raw;

        if !mag.is_finite() || mag < 0.0 {
            mag = 1e-20;
        }
        mag = mag.max(1e-20);

        if !phase.is_finite() {
            phase = 0.0;
        }

        // Store for display (WITHOUT delay applied)
        if k > 0 && f_bin <= nyq {
            interp_mag_db.push((f_bin, 20.0 * mag.log10()));
            interp_phase_deg.push((f_bin, phase_interp_raw.to_degrees()));
        }

        spectrum[k] = Complex::new(mag * phase.cos(), mag * phase.sin());
    }

    for k in 1..half {
        spectrum[fft_size - k] = spectrum[k].conj();
    }

    if fft_size % 2 == 0 {
        spectrum[half] = Complex::new(spectrum[half].re, 0.0);
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut spectrum);

    let mut ir: Vec<f64> = spectrum.iter().map(|c| c.re / fft_size as f64).collect();

    let rotation = (delay_compensation * fs) as usize;
    let ir_len = ir.len();
    if ir_len > 0 && rotation > 0 {
        ir.rotate_right(rotation % ir_len);
    }

    // Resample interpolated data for display (reduce point count)
    let interp_mag_db = resample_plot_data(&interp_mag_db);
    let interp_phase_deg = resample_plot_data(&interp_phase_deg);

    (ir, interp_mag_db, interp_phase_deg)
}

/// Compute frequency response from impulse response
pub fn freqz(impulse: &[f64], fs: usize, n: usize) -> Vec<(f64, f64, f64)> {
    let fft_size = n.next_power_of_two();
    let impulse = if impulse.len() > fft_size {
        &impulse[..fft_size]
    } else {
        impulse
    };

    let mut padded_impulse = vec![0.0; fft_size];
    padded_impulse[..impulse.len()].copy_from_slice(impulse);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut buffer: Vec<Complex<f64>> = padded_impulse
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();

    fft.process(&mut buffer);

    let nyquist = fs as f64 / 2.0;
    let half_n = fft_size / 2 + 1;

    let mut phases: Vec<f64> = Vec::with_capacity(half_n);
    let mut magnitudes: Vec<f64> = Vec::with_capacity(half_n);

    for i in 0..half_n {
        magnitudes.push(buffer[i].norm());
        phases.push(buffer[i].arg());
    }

    let unwrapped_phases = unwrap_phase_radians(&phases);

    let result: Vec<(f64, f64, f64)> = (0..half_n)
        .map(|i| {
            let freq = i as f64 * nyquist / (fft_size as f64 / 2.0);
            let mag_db = 20.0 * magnitudes[i].log10();
            let phase_deg = unwrapped_phases[i].to_degrees();
            (freq, mag_db, phase_deg)
        })
        .collect();

    // Resample for display (keep only magnitude/phase pairs for resampling)
    let mag_pairs: Vec<(f64, f64)> = result.iter().map(|(f, m, _)| (*f, *m)).collect();
    let phase_pairs: Vec<(f64, f64)> = result.iter().map(|(f, _, p)| (*f, *p)).collect();

    let resampled_mag = resample_plot_data(&mag_pairs);
    let resampled_phase = resample_plot_data(&phase_pairs);

    // Merge back: use frequencies from resampled_mag and find corresponding phase
    resampled_mag
        .into_iter()
        .map(|(f, m)| {
            // Find matching phase for this frequency
            let phase = resampled_phase
                .iter()
                .find(|(fp, _)| (*fp - f).abs() < 1e-6)
                .map(|(_, p)| *p)
                .unwrap_or(0.0);
            (f, m, phase)
        })
        .collect()
}

/// Convert an impulse response to minimum phase using the Hilbert transform method
/// This computes the minimum phase equivalent that has the same magnitude response
pub fn minimum_phase_transform(impulse: &[f64]) -> Vec<f64> {
    let n = impulse.len();
    let fft_size = n.next_power_of_two() * 2; // Use 2x for better precision

    // Zero-pad to power of 2
    let mut padded: Vec<Complex<f64>> = impulse
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    // Forward FFT to get frequency domain
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut padded);

    // Compute log magnitude spectrum (real only, no imaginary part yet)
    let mut log_mag: Vec<Complex<f64>> = padded
        .iter()
        .map(|c| {
            let mag = c.norm().max(1e-20);
            Complex::new(mag.ln(), 0.0)
        })
        .collect();

    // Apply inverse FFT to get real cepstrum
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut log_mag);

    // Scale by FFT size
    for c in log_mag.iter_mut() {
        *c = *c / fft_size as f64;
    }

    // Apply Hilbert transform window in cepstral domain
    // Keep DC, double positive quefrencies, zero negative quefrencies
    // This creates the minimum phase cepstrum
    log_mag[0] = log_mag[0]; // DC: keep as is

    for i in 1..fft_size / 2 {
        log_mag[i] = log_mag[i] * 2.0; // Double positive quefrencies
    }

    if fft_size % 2 == 0 {
        log_mag[fft_size / 2] = log_mag[fft_size / 2]; // Nyquist: keep as is
    }

    for i in (fft_size / 2 + 1)..fft_size {
        log_mag[i] = Complex::new(0.0, 0.0); // Zero negative quefrencies
    }

    // Forward FFT to get complex log spectrum (log magnitude + j*minimum phase)
    let fft2 = planner.plan_fft_forward(fft_size);
    fft2.process(&mut log_mag);

    // Convert from complex log spectrum to linear spectrum
    // exp(log|H| + j*phi) = |H| * exp(j*phi)
    let mut min_phase_spectrum: Vec<Complex<f64>> = log_mag.iter().map(|c| c.exp()).collect();

    // Inverse FFT to get minimum phase impulse response
    let ifft2 = planner.plan_fft_inverse(fft_size);
    ifft2.process(&mut min_phase_spectrum);

    // Extract real part and normalize
    min_phase_spectrum
        .iter()
        .take(n)
        .map(|c| c.re / fft_size as f64)
        .collect()
}

/// Compute pre-ringing metrics for an impulse response
/// Returns (pre_energy_percent, peak_index, pre_duration_ms, centroid_shift_ms)
pub fn compute_preringing_metrics(ir: &[f64], sample_rate: f64) -> (f64, usize, f64, f64) {
    if ir.is_empty() {
        return (0.0, 0, 0.0, 0.0);
    }

    // Add 100ms of delay by rotating to ensure pre-ringing is not truncated
    let delay_samples = (0.1 * sample_rate) as usize; // 100ms
    let mut ir_delayed = ir.to_vec();
    if delay_samples < ir_delayed.len() {
        ir_delayed.rotate_right(delay_samples);
    }

    // Find the main peak index (maximum absolute value)
    let mut peak_idx = 0;
    let mut peak_val = 0.0;
    for (i, &val) in ir_delayed.iter().enumerate() {
        let abs_val = val.abs();
        if abs_val > peak_val {
            peak_val = abs_val;
            peak_idx = i;
        }
    }

    // Calculate energy before and after peak
    let energy: Vec<f64> = ir_delayed.iter().map(|&x| x * x).collect();
    let e_pre: f64 = energy[..peak_idx].iter().sum();
    let e_total: f64 = energy.iter().sum::<f64>().max(1e-30);
    let pre_energy_percent = 100.0 * e_pre / e_total;

    // Calculate pre-duration: from first significant sample to peak
    // Use dynamic threshold: -60 dB below peak
    let dynamic_floor = 20.0 * (peak_val / 1000.0).log10(); // -60 dB below peak
    let mut start_idx = 0;
    for (i, &val) in ir_delayed[..peak_idx].iter().enumerate() {
        let mag_db = 20.0 * (val.abs() + 1e-30).log10();
        if mag_db > dynamic_floor {
            start_idx = i;
            break;
        }
    }
    let pre_duration_ms = ((peak_idx - start_idx) as f64 / sample_rate) * 1000.0;

    // Calculate time centroid (energy-weighted)
    let mut weighted_sum = 0.0;
    for (i, &e) in energy.iter().enumerate() {
        weighted_sum += (i as f64) * e;
    }
    let centroid_samples = weighted_sum / e_total;

    // For centroid shift, we would need the minimum phase version
    // For now, we'll compute a simplified metric: distance from peak to centroid
    let centroid_shift_ms = ((centroid_samples - peak_idx as f64).abs() / sample_rate) * 1000.0;

    (
        pre_energy_percent,
        peak_idx,
        pre_duration_ms,
        centroid_shift_ms,
    )
}
