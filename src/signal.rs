use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

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

/// Reconstruct impulse response from FRD data with adjustable delay
pub fn impulse_from_frd(
    frd: &[(f64, f64, f64)],
    fs: f64,
    delay_ms: f64,
    dc_extrap_rolloff: bool,
    hf_extrap_rolloff: bool,
) -> (Vec<f64>, Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let nyq = fs / 2.0;

    let mut data: Vec<(f64, f64, f64)> = frd
        .iter()
        .cloned()
        .filter(|(f, _, _)| *f > 0.0 && *f <= nyq && f.is_finite())
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

    let fft_size = (fs as usize).next_power_of_two();
    let half = fft_size / 2;

    let mut spectrum: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];

    // Store interpolated data for display
    let mut interp_mag_db = Vec::new();
    let mut interp_phase_deg = Vec::new();

    for k in 0..=half {
        let f_bin = k as f64 * fs / fft_size as f64;

        let mut mag =
            mag_interp.interpolate_with_extrap_modes(f_bin, dc_extrap_rolloff, hf_extrap_rolloff);
        let phase_interp_raw = phase_interp.interpolate_with_extrapolation(f_bin, false);
        let mut phase = phase_interp_raw - 2.0 * PI * f_bin * delay_compensation;

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

    (0..half_n)
        .map(|i| {
            let freq = i as f64 * nyquist / (fft_size as f64 / 2.0);
            let mag_db = 20.0 * magnitudes[i].log10();
            let phase_deg = unwrapped_phases[i].to_degrees();
            (freq, mag_db, phase_deg)
        })
        .collect()
}
