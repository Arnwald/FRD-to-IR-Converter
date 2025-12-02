mod signal;

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use signal::{freqz, impulse_from_frd, parse_frd};

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

    // Parameters (these will be saved)
    sample_rate: f64,
    delay_ms: f64,
    freq_min: f64,
    freq_max: f64,

    // Extrapolation modes
    dc_extrap_rolloff: bool, // true = roll-off, false = constant
    hf_extrap_rolloff: bool, // true = roll-off, false = constant

    // UI state
    wrap_phase: bool,
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
            sample_rate: 96000.0,
            delay_ms: 5.0,
            freq_min: 1.0,
            freq_max: 22000.0,
            dc_extrap_rolloff: true,
            hf_extrap_rolloff: false,
            wrap_phase: false,
            file_name: String::new(),
        }
    }
}

impl FrdToIrApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }

    fn import_frd(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("FRD Files", &["frd", "FRD", "txt", "TXT"])
            .pick_file()
        {
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

        // Convert FRD to IR with current parameters
        let (mut ir, interp_mag, interp_phase) = impulse_from_frd(
            &self.frd_data,
            self.sample_rate,
            self.delay_ms,
            self.dc_extrap_rolloff,
            self.hf_extrap_rolloff,
        );

        // Truncate IR to 2 seconds max to avoid slowing down the interface
        let max_samples = (2.0 * self.sample_rate) as usize;
        if ir.len() > max_samples {
            ir.truncate(max_samples);
        }

        self.ir_data = ir;
        self.interp_mag_db = interp_mag;
        self.interp_phase_deg = interp_phase;

        // Compute frequency response from IR
        if !self.ir_data.is_empty() {
            let freqz_data = freqz(&self.ir_data, self.sample_rate as usize, 8192);
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
            }

            ui.separator();

            // Parameters
            ui.horizontal(|ui| {
                ui.label("Sample Rate:");
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
                    });

                ui.add_space(20.0);

                ui.label("Delay:");
                let delay_changed = ui
                    .add(
                        egui::DragValue::new(&mut self.delay_ms)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix(" ms"),
                    )
                    .changed();

                ui.add_space(20.0);

                if ui
                    .toggle_value(&mut self.wrap_phase, "Wrap Phase")
                    .changed()
                {
                    // No need to reconvert, just affects display
                }

                ui.add_space(20.0);

                ui.label("DC Extrap:");
                let dc_changed = egui::ComboBox::from_id_salt("dc_extrap")
                    .selected_text(if self.dc_extrap_rolloff {
                        "Roll-off"
                    } else {
                        "Constant"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.dc_extrap_rolloff, false, "Constant")
                            .clicked()
                            || ui
                                .selectable_value(&mut self.dc_extrap_rolloff, true, "Roll-off")
                                .clicked()
                    })
                    .inner
                    .unwrap_or(false);

                ui.add_space(10.0);

                ui.label("HF Extrap:");
                let hf_changed = egui::ComboBox::from_id_salt("hf_extrap")
                    .selected_text(if self.hf_extrap_rolloff {
                        "Roll-off"
                    } else {
                        "Constant"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.hf_extrap_rolloff, false, "Constant")
                            .clicked()
                            || ui
                                .selectable_value(&mut self.hf_extrap_rolloff, true, "Roll-off")
                                .clicked()
                    })
                    .inner
                    .unwrap_or(false);

                ui.add_space(20.0);

                ui.label("Freq Min:");
                ui.add(
                    egui::DragValue::new(&mut self.freq_min)
                        .speed(1.0)
                        .range(1.0..=self.freq_max - 1.0)
                        .suffix(" Hz"),
                );

                ui.add_space(10.0);

                ui.label("Freq Max:");
                ui.add(
                    egui::DragValue::new(&mut self.freq_max)
                        .speed(10.0)
                        .range(self.freq_min + 1.0..=48000.0)
                        .suffix(" Hz"),
                );

                if sr_changed || delay_changed || dc_changed || hf_changed {
                    self.update_conversion();
                }
            });

            ui.separator();

            if self.frd_data.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(200.0);
                    ui.heading("No FRD file loaded");
                    ui.label("Use File → Open FRD... to load a file");
                });
                return;
            }

            // Three graphs in a column
            let available_height = ui.available_height();
            // Reserve space for separators (2 × ~20px) and margins
            let graph_height = (available_height - 100.0) / 3.0;

            // Helper functions for log axis
            let log_spacer = |input: egui_plot::GridInput| -> Vec<egui_plot::GridMark> {
                let (min, max) = input.bounds;
                let mut marks = Vec::new();
                let min_pow = min.log10().floor() as i32;
                let max_pow = max.log10().ceil() as i32;

                for pow in min_pow..=max_pow {
                    let base = 10f64.powi(pow);
                    // Place marks at 1, 2, and 5 of each decade
                    for m in [1.0, 2.0, 5.0] {
                        let value = base * m;
                        if value >= min && value <= max {
                            marks.push(egui_plot::GridMark {
                                value,
                                step_size: base,
                            });
                        }
                    }
                }
                marks
            };

            let log_formatter =
                |mark: egui_plot::GridMark, _range: &std::ops::RangeInclusive<f64>| -> String {
                    let freq = mark.value;
                    if freq >= 1000.0 {
                        format!("{:.0}k", freq / 1000.0)
                    } else {
                        format!("{:.0}", freq)
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
                        // Truncate display to 100ms max for clarity
                        let max_display_samples =
                            ((0.1 * self.sample_rate) as usize).min(self.ir_data.len());
                        let ir_points: PlotPoints = self
                            .ir_data
                            .iter()
                            .take(max_display_samples)
                            .enumerate()
                            .map(|(i, v)| {
                                let t_ms = i as f64 * 1000.0 / self.sample_rate;
                                [t_ms, *v]
                            })
                            .collect();
                        plot_ui.line(Line::new("Impulse Response", ir_points));
                    }
                });

            ui.separator();

            // Graph 3: Reconstructed Frequency Response (from IR)
            ui.label(egui::RichText::new("Reconstructed Frequency Response (from IR)").strong());

            let recon_phase_data = if self.wrap_phase {
                Self::wrap_phase_data(&self.reconstructed_phase_deg)
            } else {
                self.reconstructed_phase_deg.clone()
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
    }
}
