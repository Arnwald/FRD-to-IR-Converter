mod signal;

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use signal::{freqz, impulse_from_frd, parse_frd, compute_preringing_metrics};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("FRD to IR Converter"),
        ..Default::default()
    };

    eframe::run_native(
        "FRD to IR Converter",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(FrdToIrApp::new(cc)))
        }),
    )
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
struct FrdToIrApp {
    // FRD data
    frd_data: Vec<(f64, f64, f64)>, // (freq, mag_db, phase_deg)

    // Interpolated data
    interp_mag_db: Vec<(f64, f64)>,
    interp_phase_deg: Vec<(f64, f64)>,

    // IR data
    ir_data: Vec<f64>,

    // Reconstructed frequency response
    reconstructed_mag_db: Vec<(f64, f64)>,
    reconstructed_phase_deg: Vec<(f64, f64)>,

    // Data quality metrics
    frd_point_count: usize,
    expected_point_count: usize,
    data_coverage_percent: f64,
    has_dc: bool,
    has_nyquist: bool,
    preringing_percent: f64,
    preringing_duration_ms: f64,

    // Parameters (these will be saved)
    sample_rate: f64,
    delay_ms: f64,
    freq_min: f64,
    freq_max: f64,
    ir_start_ms: f64,
    ir_stop_ms: f64,

    // Processing options
    minimum_phase: bool,
    normalize_on_export: bool,

    // UI state
    wrap_phase: bool,
    remove_delay_phase: bool,
    show_filter_window: bool,
    show_data_info_window: bool,
    show_export_dialog: bool,
    show_normalization_popup: bool,
    normalization_factor: f64,
    file_path: String,
    #[serde(skip)]
    file_name: String,
}

impl Default for FrdToIrApp {
    fn default() -> Self {
        Self {
            frd_data: Vec::new(),
            interp_mag_db: Vec::new(),
            interp_phase_deg: Vec::new(),
            ir_data: Vec::new(),
            reconstructed_mag_db: Vec::new(),
            reconstructed_phase_deg: Vec::new(),
            frd_point_count: 0,
            expected_point_count: 0,
            data_coverage_percent: 0.0,
            has_dc: false,
            has_nyquist: false,
            preringing_percent: 0.0,
            preringing_duration_ms: 0.0,
            sample_rate: 96000.0,
            delay_ms: 5.0,
            freq_min: 1.0,
            freq_max: 22000.0,
            ir_start_ms: 0.0,
            ir_stop_ms: 100.0,
            minimum_phase: false,
            normalize_on_export: false,
            wrap_phase: false,
            remove_delay_phase: false,
            show_filter_window: false,
            show_data_info_window: false,
            show_export_dialog: false,
            show_normalization_popup: false,
            normalization_factor: 1.0,
            file_path: String::new(),
            file_name: String::new(),
        }
    }
}

impl FrdToIrApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            let mut app: Self = eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            // Reload FRD file if we have a saved path
            if !app.file_path.is_empty() {
                if let Ok(text) = std::fs::read_to_string(&app.file_path) {
                    app.frd_data = parse_frd(&text);
                    // Restore file name from path
                    app.file_name = std::path::Path::new(&app.file_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("Unknown")
                        .to_string();
                    if !app.frd_data.is_empty() {
                        app.update_conversion();
                    }
                }
            }
            return app;
        }

        Default::default()
    }

    fn import_frd(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("FRD Files", &["frd", "FRD", "txt", "TXT"])
            .pick_file()
        {
            self.file_path = path.to_string_lossy().to_string();
            self.file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .to_string();

            if let Ok(text) = std::fs::read_to_string(&path) {
                self.frd_data = parse_frd(&text);
                if !self.frd_data.is_empty() {
                    self.update_conversion();
                }
            }
        }
    }

    fn update_conversion(&mut self) {
        if self.frd_data.is_empty() {
            return;
        }

        // Calculate data quality metrics
        let nyquist = self.sample_rate / 2.0;
        let fft_size = (self.sample_rate as usize).next_power_of_two();
        let freq_resolution = self.sample_rate / fft_size as f64;
        
        // Check for DC (0 Hz) and Nyquist frequency
        self.has_dc = self.frd_data.iter().any(|(f, _, _)| *f == 0.0);
        self.has_nyquist = self.frd_data.iter().any(|(f, _, _)| (*f - nyquist).abs() < freq_resolution);
        
        // Count actual FRD points within Nyquist range
        self.frd_point_count = self.frd_data.iter()
            .filter(|(f, _, _)| *f > 0.0 && *f <= nyquist)
            .count();
        
        // Calculate expected number of frequency bins from DC to Nyquist
        self.expected_point_count = (nyquist / freq_resolution) as usize;
        
        // Calculate coverage percentage
        self.data_coverage_percent = if self.expected_point_count > 0 {
            (self.frd_point_count as f64 / self.expected_point_count as f64) * 100.0
        } else {
            0.0
        };

        // Convert FRD to IR with current parameters
        let (ir, interp_mag, interp_phase) =
            impulse_from_frd(&self.frd_data, self.sample_rate, self.delay_ms);

        // Compute pre-ringing metrics on the original IR (before minimum phase)
        let (pre_percent, _peak_idx, pre_duration, _centroid_shift) = 
            compute_preringing_metrics(&ir, self.sample_rate);
        self.preringing_percent = pre_percent;
        self.preringing_duration_ms = pre_duration;

        // Apply minimum phase transformation if requested
        let ir_final = if self.minimum_phase && !ir.is_empty() {
            // If pre-ringing is detected and would be truncated, apply extra delay before Hilbert transform
            let required_delay_ms = pre_duration * 1.5;
            let needs_extra_delay = required_delay_ms > self.delay_ms;
            
            let ir_for_hilbert = if needs_extra_delay {
                // Apply temporary extra delay to avoid truncating pre-ringing
                let extra_delay_samples = (required_delay_ms * self.sample_rate / 1000.0) as usize;
                let mut ir_temp = ir.clone();
                if extra_delay_samples > 0 && extra_delay_samples < ir_temp.len() {
                    ir_temp.rotate_right(extra_delay_samples);
                }
                ir_temp
            } else {
                ir.clone()
            };
            
            // Apply minimum phase transformation
            let ir_min_phase = signal::minimum_phase_transform(&ir_for_hilbert);

            // Apply the user-requested delay by rotating the IR
            let delay_samples = (self.delay_ms * self.sample_rate / 1000.0) as usize;
            let mut ir_delayed = ir_min_phase;
            if delay_samples > 0 && delay_samples < ir_delayed.len() {
                ir_delayed.rotate_right(delay_samples);
            }
            ir_delayed
        } else {
            ir
        };

        self.ir_data = ir_final.clone();
        self.interp_mag_db = interp_mag;
        self.interp_phase_deg = interp_phase;

        // Compute frequency response from IR
        if !ir_final.is_empty() {
            let freqz_data = freqz(&ir_final, self.sample_rate as usize, (1 << 17) as usize);
            self.reconstructed_mag_db = freqz_data.iter().map(|(f, m, _)| (*f, *m)).collect();
            self.reconstructed_phase_deg = freqz_data.iter().map(|(f, _, p)| (*f, *p)).collect();
        }
    }

    fn wrap_phase_data(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
        data.iter()
            .map(|(f, p)| {
                // Robust phase wrapping to [-180, 180]
                let mut wrapped = p % 360.0;
                if wrapped > 180.0 {
                    wrapped -= 360.0;
                } else if wrapped < -180.0 {
                    wrapped += 360.0;
                }
                (*f, wrapped)
            })
            .collect()
    }

    fn unwrap_phase_degrees(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
        if data.is_empty() {
            return vec![];
        }

        let mut unwrapped = Vec::with_capacity(data.len());
        unwrapped.push(data[0]);
        let mut cumulative_offset = 0.0;

        for i in 1..data.len() {
            let diff = data[i].1 - data[i - 1].1;

            if diff > 180.0 {
                cumulative_offset -= 360.0;
            } else if diff < -180.0 {
                cumulative_offset += 360.0;
            }

            unwrapped.push((data[i].0, data[i].1 + cumulative_offset));
        }

        unwrapped
    }

    fn remove_linear_phase(data: &[(f64, f64)], delay_ms: f64) -> Vec<(f64, f64)> {
        // Remove the linear phase component corresponding to the delay
        // Phase shift (in degrees) = -360 * frequency * delay
        data.iter()
            .map(|(f, p)| {
                let delay_phase_deg = -360.0 * f * delay_ms / 1000.0;
                (*f, p - delay_phase_deg)
            })
            .collect()
    }

    fn export_wav(&mut self) {
        if self.ir_data.is_empty() {
            return;
        }

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("WAV Files", &["wav", "WAV"])
            .set_file_name("impulse_response.wav")
            .save_file()
        {
            // Get peak level
            let max_val = self.ir_data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            
            // Apply normalization if user requested it
            let (ir_to_export, was_normalized): (Vec<f32>, bool) = if self.normalize_on_export && max_val > 0.0 {
                let normalized = self.ir_data.iter().map(|&x| (x / max_val) as f32).collect();
                (normalized, true)
            } else {
                let as_f32 = self.ir_data.iter().map(|&x| x as f32).collect();
                (as_f32, false)
            };

            // Write WAV file
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: self.sample_rate as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            if let Ok(mut writer) = hound::WavWriter::create(&path, spec) {
                for &sample in &ir_to_export {
                    let _ = writer.write_sample(sample);
                }
                let _ = writer.finalize();
                
                // Show popup if normalization was applied
                if was_normalized {
                    self.normalization_factor = max_val;
                    self.show_normalization_popup = true;
                }
            }
        }
    }
}

impl eframe::App for FrdToIrApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open FRD...").clicked() {
                        self.import_frd();
                        ui.close();
                    }
                    
                    if ui.button("Export WAV...").on_hover_text(
                        "Export impulse response to WAV file (32-bit float).\n\n\
                        The exported file can contain values > 1.0 without issue.\n\
                        You can choose to normalize to 1.0 if needed for compatibility."
                    ).clicked() {
                        self.show_export_dialog = true;
                        ui.close();
                    }
                    
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.add_space(16.0);
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        // Main panel
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("FRD to IR Converter");

            if !self.file_name.is_empty() {
                ui.label(format!("Loaded: {}", self.file_name));
                
                // Display data quality information
                if self.frd_point_count > 0 {
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Data Quality: {} / {} frequency points ({:.1}% coverage)",
                            self.frd_point_count,
                            self.expected_point_count,
                            self.data_coverage_percent
                        ))
                        .on_hover_text(
                            "FRD files typically contain sparse frequency data.\n\
                            This shows how many data points are present vs. the number needed\n\
                            for full frequency resolution at the selected sample rate.\n\
                            Lower coverage means more interpolation is required."
                        );
                        
                        // Color-code the quality indicator
                        let quality_color = if self.data_coverage_percent >= 50.0 {
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::from_rgb(0, 120, 0) // Much darker green for light mode
                            }
                        } else if self.data_coverage_percent >= 20.0 {
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::from_rgb(180, 120, 0)
                            }
                        } else {
                            egui::Color32::RED
                        };
                        
                        ui.colored_label(
                            quality_color,
                            if self.data_coverage_percent >= 50.0 {
                                "Good"
                            } else if self.data_coverage_percent >= 20.0 {
                                "Moderate"
                            } else {
                                "Low"
                            }
                        );
                        
                        // Info button to show detailed data issues
                        if ui.button("Info").clicked() {
                            self.show_data_info_window = true;
                        }
                    });
                }
            }

            ui.separator();

            // Parameters
            ui.horizontal(|ui| {
                ui.label("Sample Rate:").on_hover_text("Sampling rate of the generated impulse response.");
                let mut sr_changed = false;
                egui::ComboBox::from_id_salt("sample_rate")
                    .selected_text(format!("{} Hz", self.sample_rate as u32))
                    .show_ui(ui, |ui| {
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 44100.0, "44100 Hz")
                            .changed();
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 48000.0, "48000 Hz")
                            .changed();
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 88200.0, "88200 Hz")
                            .changed();
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 96000.0, "96000 Hz")
                            .changed();
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 176400.0, "176400 Hz")
                            .changed();
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 192000.0, "192000 Hz")
                            .changed();
                        sr_changed |= ui
                            .selectable_value(&mut self.sample_rate, 384000.0, "384000 Hz")
                            .changed();
                    });

                ui.add_space(20.0);

                ui.label("Delay:").on_hover_text("Delay applied to the impulse to avoid time-domain aliasing and facilitate processing.");
                let delay_changed = ui
                    .add(
                        egui::DragValue::new(&mut self.delay_ms)
                            .speed(0.1)
                            .range(0.0..=500.0)
                            .suffix(" ms"),
                    )
                    .changed();

                // Warning if pre-ringing duration exceeds delay (only if minimum phase is disabled)
                if !self.minimum_phase && self.preringing_duration_ms > self.delay_ms && self.preringing_percent > 0.1 {
                    ui.label(
                        egui::RichText::new("\u{26A0}")
                            .color(if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::from_rgb(180, 120, 0)
                            })
                    )
                    .on_hover_text(
                        format!(
                            "Pre-ringing truncation warning\n\n\
                            Detected pre-ringing duration ({:.1} ms) exceeds the current delay ({:.1} ms).\n\
                            This may truncate pre-ringing artifacts and introduce discontinuities.\n\n\
                            Recommended actions:\n\
                            • Increase delay to at least {:.1} ms, or\n\
                            • Enable 'Minimum Phase' to eliminate pre-ringing entirely",
                            self.preringing_duration_ms,
                            self.delay_ms,
                            (self.preringing_duration_ms * 1.01).ceil()
                        )
                    );
                }

                ui.add_space(20.0);

                if ui
                    .toggle_value(&mut self.wrap_phase, "Wrap Phase")
                    .on_hover_text("Wraps phase into the [-180°, 180°] range for simplified visualization.")
                    .changed()
                {
                    // No need to reconvert, just affects display
                }

                ui.add_space(20.0);

                let minimum_phase_changed = ui
                    .toggle_value(&mut self.minimum_phase, "Minimum Phase")
                    .on_hover_text(
                        "Applies a Hilbert transform to make the IR causal.\n\n\
                        Many FRD files contain acausal anomalies that cause significant pre-ringing.\n\
                        This option eliminates all pre-ringing but removes non-minimum phase behaviors\n\
                        (which are corrupted by poorly conditioned FRD files anyway)."
                    )
                    .changed();

                if sr_changed || delay_changed || minimum_phase_changed {
                    self.update_conversion();
                }
            });

            // Second row of controls
            ui.horizontal(|ui| {
                ui.label("IR Start:").on_hover_text("Start time for impulse response visualization.");
                ui.add(
                    egui::DragValue::new(&mut self.ir_start_ms)
                        .speed(0.1)
                        .range(0.0..=self.ir_stop_ms - 0.1)
                        .suffix(" ms"),
                );

                ui.add_space(10.0);

                ui.label("IR Stop:").on_hover_text("Stop time for impulse response visualization.");
                let max_ir_time_ms = (self.ir_data.len() as f64 * 1000.0) / self.sample_rate;
                ui.add(
                    egui::DragValue::new(&mut self.ir_stop_ms)
                        .speed(1.0)
                        .range(self.ir_start_ms + 0.1..=max_ir_time_ms)
                        .suffix(" ms"),
                );

                ui.add_space(20.0);

                ui.label("Freq Min:").on_hover_text("Minimum frequency for frequency response graph visualization.");
                ui.add(
                    egui::DragValue::new(&mut self.freq_min)
                        .speed(1.0)
                        .range(1.0..=self.freq_max - 1.0)
                        .suffix(" Hz"),
                );

                ui.add_space(10.0);

                ui.label("Freq Max:").on_hover_text("Maximum frequency for frequency response graph visualization.");
                ui.add(
                    egui::DragValue::new(&mut self.freq_max)
                        .speed(10.0)
                        .range(self.freq_min + 1.0..=48000.0)
                        .suffix(" Hz"),
                );

                ui.add_space(20.0);

                if ui
                    .toggle_value(&mut self.remove_delay_phase, "Remove Delay Phase")
                    .on_hover_text("Removes the linear phase component corresponding to the added delay, to visualize the intrinsic system phase (not applied to exported data).")
                    .changed()
                {
                    // No need to reconvert, just affects display
                }
            });

            ui.separator();

            if self.frd_data.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(200.0);
                    ui.heading("No FRD file loaded");
                    ui.label("Use File -> Open FRD... to load a file");
                });
                return;
            }

            // Three graphs in a column
            let available_height = ui.available_height();
            // Reserve space for separators (2 × ~20px) and margins
            let graph_height = (available_height - 100.0) / 3.0;

            // Helper functions for log axis (data is already in log10(Hz))
            // Based on linfir's implementation
            let log_spacer = |input: egui_plot::GridInput| -> Vec<egui_plot::GridMark> {
                let (min, max) = input.bounds;
                let mut marks = vec![];
                for i in min.floor() as i32..=max.ceil() as i32 {
                    marks.extend(
                        (10..100)
                            .map(|j| {
                                let value = i as f64 + (j as f64).log10() - 1.0;
                                let step_size = if j == 10 {
                                    1.0
                                } else if j % 10 == 0 {
                                    0.1
                                } else {
                                    0.01
                                };
                                egui_plot::GridMark { value, step_size }
                            })
                            .filter(|gm| (min..=max).contains(&gm.value)),
                    );
                }
                marks
            };

            let log_formatter =
                |mark: egui_plot::GridMark, _range: &std::ops::RangeInclusive<f64>| -> String {
                    let x = 10.0_f64.powf(mark.value).round();
                    match x {
                        x if x == 10.0 => "10".to_string(),
                        x if x == 20.0 => "20".to_string(),
                        x if x == 50.0 => "50".to_string(),
                        x if x == 100.0 => "100".to_string(),
                        x if x == 200.0 => "200".to_string(),
                        x if x == 500.0 => "500".to_string(),
                        x if x == 1_000.0 => "1k".to_string(),
                        x if x == 2_000.0 => "2k".to_string(),
                        x if x == 5_000.0 => "5k".to_string(),
                        x if x == 10_000.0 => "10k".to_string(),
                        x if x == 20_000.0 => "20k".to_string(),
                        _ => "".to_string(), // Hide other ticks
                    }
                };

            // Graph 1: FRD Data (Magnitude and Phase with log frequency axis)
            ui.label(egui::RichText::new("FRD Data").strong());

            // Prepare phase data
            let phase_data_raw = if self.wrap_phase {
                Self::wrap_phase_data(
                    &self
                        .frd_data
                        .iter()
                        .map(|(f, _, p)| (*f, *p))
                        .collect::<Vec<_>>(),
                )
            } else {
                // Unwrap the raw FRD phase for display consistency with interpolated phase
                Self::unwrap_phase_degrees(
                    &self
                        .frd_data
                        .iter()
                        .map(|(f, _, p)| (*f, *p))
                        .collect::<Vec<_>>(),
                )
            };

            let interp_phase_data = if self.wrap_phase {
                Self::wrap_phase_data(&self.interp_phase_deg)
            } else {
                self.interp_phase_deg.clone()
            };

            ui.horizontal(|ui| {
                // Magnitude plot with log X axis
                Plot::new("frd_mag")
                    .height(graph_height)
                    .width(ui.available_width() / 2.0 - 10.0)
                    .legend(egui_plot::Legend::default())
                    .x_axis_label("Frequency [Hz]")
                    .y_axis_label("Magnitude [dB]")
                    .x_grid_spacer(log_spacer)
                    .x_axis_formatter(log_formatter)
                    .label_formatter(|name, value| {
                        let freq_hz = 10.0_f64.powf(value.x);
                        format!("{}\n{:.0} Hz\n{:.1} dB", name, freq_hz, value.y)
                    })
                    .show(ui, |plot_ui| {
                        // Magnitude - Original data (filtered)
                        let frd_mag_points: PlotPoints = self
                            .frd_data
                            .iter()
                            .filter(|(f, _, _)| *f >= self.freq_min && *f <= self.freq_max)
                            .map(|(f, m, _)| [f.log10(), *m])
                            .collect();
                        plot_ui.line(Line::new("Original", frd_mag_points));

                        // Magnitude - Interpolated data (filtered)
                        if !self.interp_mag_db.is_empty() {
                            let interp_mag_points: PlotPoints = self
                                .interp_mag_db
                                .iter()
                                .filter(|(f, _)| *f >= self.freq_min && *f <= self.freq_max)
                                .map(|(f, m)| [f.log10(), *m])
                                .collect();
                            plot_ui.line(
                                Line::new("Interpolated", interp_mag_points)
                                    .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                            );
                        }
                    });

                // Phase plot with log X axis
                Plot::new("frd_phase")
                    .height(graph_height)
                    .width(ui.available_width() - 10.0)
                    .legend(egui_plot::Legend::default())
                    .x_axis_label("Frequency [Hz]")
                    .y_axis_label("Phase [°]")
                    .x_grid_spacer(log_spacer)
                    .x_axis_formatter(log_formatter)
                    .label_formatter(|name, value| {
                        let freq_hz = 10.0_f64.powf(value.x);
                        format!("{}\n{:.0} Hz\n{:.1}°", name, freq_hz, value.y)
                    })
                    .show(ui, |plot_ui| {
                        // Phase - Original data (filtered)
                        let frd_phase_points: PlotPoints = phase_data_raw
                            .iter()
                            .filter(|(f, _)| *f >= self.freq_min && *f <= self.freq_max)
                            .map(|(f, p)| [f.log10(), *p])
                            .collect();
                        plot_ui.line(Line::new("Raw", frd_phase_points));

                        // Phase - Interpolated with delay (filtered)
                        if !interp_phase_data.is_empty() {
                            let interp_phase_points: PlotPoints = interp_phase_data
                                .iter()
                                .filter(|(f, _)| *f >= self.freq_min && *f <= self.freq_max)
                                .map(|(f, p)| [f.log10(), *p])
                                .collect();
                            plot_ui.line(
                                Line::new("Interpolated", interp_phase_points)
                                    .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                            );
                        }
                    });
            });

            ui.separator();

            // Graph 2: Impulse Response
            ui.label(egui::RichText::new("Impulse Response").strong());

            Plot::new("ir_plot")
                .height(graph_height)
                .width(ui.available_width())
                .x_axis_label("Time [ms]")
                .y_axis_label("Amplitude")
                .label_formatter(|name, value| {
                    format!("{}\n{:.3} ms\n{:.6}", name, value.x, value.y)
                })
                .show(ui, |plot_ui| {
                    if !self.ir_data.is_empty() {
                        // Display IR according to start/stop time range
                        let start_sample = (self.ir_start_ms * self.sample_rate / 1000.0) as usize;
                        let stop_sample = ((self.ir_stop_ms * self.sample_rate / 1000.0) as usize)
                            .min(self.ir_data.len());

                        if start_sample < stop_sample {
                            let ir_points: PlotPoints = self
                                .ir_data
                                .iter()
                                .enumerate()
                                .skip(start_sample)
                                .take(stop_sample - start_sample)
                                .map(|(i, v)| {
                                    let t_ms = i as f64 * 1000.0 / self.sample_rate;
                                    [t_ms, *v]
                                })
                                .collect();
                            plot_ui.line(Line::new("Impulse Response", ir_points));
                        }
                    }
                });

            ui.separator();

            // Graph 3: Reconstructed Frequency Response (from IR)
            ui.label(egui::RichText::new("Reconstructed Frequency Response (from IR)").strong());

            // Apply delay phase removal if requested
            let recon_phase_data = if self.remove_delay_phase {
                Self::remove_linear_phase(&self.reconstructed_phase_deg, self.delay_ms)
            } else {
                self.reconstructed_phase_deg.clone()
            };

            let recon_phase_data = if self.wrap_phase {
                Self::wrap_phase_data(&recon_phase_data)
            } else {
                recon_phase_data
            };

            ui.horizontal(|ui| {
                // Magnitude plot with log X axis
                Plot::new("recon_mag")
                    .height(graph_height)
                    .width(ui.available_width() / 2.0 - 10.0)
                    .x_axis_label("Frequency [Hz]")
                    .y_axis_label("Magnitude [dB]")
                    .x_grid_spacer(log_spacer)
                    .x_axis_formatter(log_formatter)
                    .label_formatter(|name, value| {
                        let freq_hz = 10.0_f64.powf(value.x);
                        format!("{}\n{:.0} Hz\n{:.1} dB", name, freq_hz, value.y)
                    })
                    .show(ui, |plot_ui| {
                        if !self.reconstructed_mag_db.is_empty() {
                            let recon_mag_points: PlotPoints = self
                                .reconstructed_mag_db
                                .iter()
                                .filter(|(f, _)| {
                                    *f > 0.0 && *f >= self.freq_min && *f <= self.freq_max
                                })
                                .map(|(f, m)| [f.log10(), *m])
                                .collect();
                            plot_ui.line(Line::new("Reconstructed", recon_mag_points));
                        }
                    });

                // Phase plot with log X axis
                Plot::new("recon_phase")
                    .height(graph_height)
                    .width(ui.available_width() - 10.0)
                    .x_axis_label("Frequency [Hz]")
                    .y_axis_label("Phase [°]")
                    .x_grid_spacer(log_spacer)
                    .x_axis_formatter(log_formatter)
                    .label_formatter(|name, value| {
                        let freq_hz = 10.0_f64.powf(value.x);
                        format!("{}\n{:.0} Hz\n{:.1}°", name, freq_hz, value.y)
                    })
                    .show(ui, |plot_ui| {
                        if !recon_phase_data.is_empty() {
                            let recon_phase_points: PlotPoints = recon_phase_data
                                .iter()
                                .filter(|(f, _)| {
                                    *f > 0.0 && *f >= self.freq_min && *f <= self.freq_max
                                })
                                .map(|(f, p)| [f.log10(), *p])
                                .collect();
                            plot_ui.line(Line::new("Reconstructed", recon_phase_points));
                        }
                    });
            });
        });
        
        // Data Info Window
        if self.show_data_info_window {
            egui::Window::new("Data Quality Information")
                .open(&mut self.show_data_info_window)
                .resizable(true)
                .default_width(500.0)
                .show(ctx, |ui| {
                    ui.heading("Issues Detected in FRD File");
                    ui.separator();
                    
                    let mut has_issues = false;
                    
                    // Check for missing DC
                    if !self.has_dc {
                        has_issues = true;
                        ui.label(egui::RichText::new("\u{26A0} Missing DC (0 Hz) data point").color(
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::from_rgb(180, 120, 0)
                            }
                        ));
                        ui.label("   The FRD file does not contain a measurement at 0 Hz.");
                        ui.add_space(10.0);
                    }
                    
                    // Check for missing Nyquist
                    if !self.has_nyquist {
                        has_issues = true;
                        let nyquist = self.sample_rate / 2.0;
                        ui.label(egui::RichText::new(
                            format!("\u{26A0} Missing data near Nyquist frequency ({:.0} Hz)", nyquist)
                        ).color(
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::from_rgb(180, 120, 0)
                            }
                        ));
                        ui.label(format!("   The FRD file does not contain measurements up to {:.0} Hz.", nyquist));
                        ui.add_space(10.0);
                    }
                    
                    // Check for sparse data
                    if self.data_coverage_percent < 50.0 {
                        has_issues = true;
                        let missing_percent = 100.0 - self.data_coverage_percent;
                        ui.label(egui::RichText::new(
                            format!("\u{26A0} Sparse frequency data ({:.1}% missing)", missing_percent)
                        ).color(egui::Color32::RED));
                        ui.label(format!(
                            "   Only {} out of {} expected frequency points are present.",
                            self.frd_point_count, self.expected_point_count
                        ));
                        ui.label(format!(
                            "   {:.1}% of the frequency data is missing and will be interpolated.",
                            missing_percent
                        ));
                        ui.add_space(10.0);
                    }
                    
                    // Check for pre-ringing
                    if self.preringing_percent > 0.1 {
                        has_issues = true;
                        let severity_color = if self.preringing_percent >= 1.0 {
                            egui::Color32::RED
                        } else if self.preringing_percent >= 0.5 {
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::from_rgb(180, 120, 0)
                            }
                        } else {
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::from_rgb(100, 200, 255)
                            } else {
                                egui::Color32::from_rgb(0, 100, 180)
                            }
                        };
                        
                        let severity_text = if self.preringing_percent >= 1.0 {
                            "\u{26A0} Significant pre-ringing detected"
                        } else if self.preringing_percent >= 0.5 {
                            "\u{26A0} Moderate pre-ringing detected"
                        } else {
                            "\u{26A0} Minor pre-ringing detected"
                        };
                        
                        ui.label(egui::RichText::new(
                            format!("{} ({:.2}% energy before main peak)", severity_text, self.preringing_percent)
                        ).color(severity_color));
                        ui.label(format!(
                            "   Pre-ring duration: {:.2} ms",
                            self.preringing_duration_ms
                        ));
                        ui.label(
                            "   This indicates the phase response may have been altered\n\
                             (smoothing, wrapping errors, or measurement artifacts)."
                        );
                        if self.preringing_percent >= 1.0 {
                            ui.label(
                                "   This level of pre-ringing will affect transient response and introduce errors with phase correction.\n\
                                 Consider using 'Minimum Phase' option to eliminate pre-ringing."
                            );
                        }
                        ui.add_space(10.0);
                    }
                    
                    if !has_issues {
                        ui.label(egui::RichText::new("\u{2713} No major issues detected").color(
                            if ui.ctx().style().visuals.dark_mode {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::from_rgb(0, 120, 0)
                            }
                        ));
                        ui.label("   The FRD file contains DC, data up to Nyquist, and sufficient frequency points.");
                    }
                    
                    ui.separator();
                    ui.heading("Impact on Impulse Response");
                    ui.add_space(5.0);
                    
                    ui.label(egui::RichText::new("\u{26A0} Warning").color(egui::Color32::from_rgb(255, 150, 0)));
                    ui.add_space(5.0);
                    
                    ui.label(
                        "Missing DC data, missing Nyquist data, or sparse frequency coverage can cause\n\
                        artifacts in the reconstructed impulse response."
                    );
                    ui.add_space(5.0);
                    
                    ui.label(
                        "The final impulse may not be fully representative of the original system,\n\
                        as missing information has been estimated through interpolation and extrapolation."
                    );
                    ui.add_space(5.0);
                    
                    ui.label(
                        "For best results, use FRD files with:\n\
                        • A data point at DC (0 Hz)\n\
                        • Data extending to at least the Nyquist frequency\n\
                        • Dense frequency spacing (> 50% coverage)\n\
                        • Linear frequency spacing if possible"
                    );
                });
        }
        
        // Export dialog
        if self.show_export_dialog {
            let mut close_dialog = false;
            let mut do_export = false;
            
            egui::Window::new("Export WAV")
                .resizable(false)
                .collapsible(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.heading("Export Options");
                    ui.separator();
                    ui.add_space(5.0);
                    
                    // Show peak level info
                    if !self.ir_data.is_empty() {
                        let max_val = self.ir_data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
                        ui.label(format!("Current peak level: {:.6} ({:.2} dB)", max_val, 20.0 * max_val.log10()));
                        
                        if max_val > 1.0 {
                            ui.colored_label(
                                if ui.ctx().style().visuals.dark_mode {
                                    egui::Color32::YELLOW
                                } else {
                                    egui::Color32::from_rgb(180, 120, 0)
                                },
                                format!("Peak exceeds 1.0 by {:.2} dB", 20.0 * max_val.log10())
                            );
                        } else {
                            ui.colored_label(
                                if ui.ctx().style().visuals.dark_mode {
                                    egui::Color32::GREEN
                                } else {
                                    egui::Color32::from_rgb(0, 120, 0)
                                },
                                "Peak is within [0.0, 1.0] range"
                            );
                        }
                    }
                    
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(5.0);
                    
                    // Normalization option
                    ui.checkbox(&mut self.normalize_on_export, "Normalize to 1.0")
                        .on_hover_text(
                            "Divide all samples by the peak value to ensure the maximum\n\
                            absolute value is 1.0. This improves compatibility with some software\n\
                            but reduces headroom.\n\n\
                            Note: 32-bit float WAV files can safely store values > 1.0,\n\
                            so normalization is usually not necessary for modern DAWs."
                        );
                    
                    ui.add_space(15.0);
                    ui.separator();
                    
                    // Buttons
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            close_dialog = true;
                        }
                        
                        ui.add_space(10.0);
                        
                        if ui.button("Export").clicked() {
                            do_export = true;
                            close_dialog = true;
                        }
                    });
                });
            
            if close_dialog {
                self.show_export_dialog = false;
            }
            
            if do_export {
                self.export_wav();
            }
        }
        
        // Normalization popup
        if self.show_normalization_popup {
            let mut close_popup = false;
            egui::Window::new("Normalization Applied")
                .resizable(false)
                .collapsible(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.heading("WAV Export - Normalization Applied");
                    ui.separator();
                    ui.add_space(5.0);
                    
                    ui.label(
                        format!(
                            "The impulse response peak level ({:.2} dB / {:.6} linear) exceeded 1.0,\n\
                            so the signal was normalized to prevent clipping in the WAV file.",
                            20.0 * self.normalization_factor.log10(),
                            self.normalization_factor
                        )
                    );
                    ui.add_space(10.0);
                    
                    ui.label(
                        format!("Normalization factor applied: {:.6} (divided by {:.6})",
                            1.0 / self.normalization_factor,
                            self.normalization_factor
                        )
                    );
                    ui.add_space(10.0);
                    
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.add_space(ui.available_width() / 2.0 - 30.0);
                        if ui.button("OK").clicked() {
                            close_popup = true;
                        }
                    });
                });
            
            if close_popup {
                self.show_normalization_popup = false;
            }
        }
    }
}
