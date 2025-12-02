# FRD to IR Converter

A simple Rust application to convert FRD (Frequency Response Data) files to Impulse Responses.

## Features

- Import FRD files (frequency, magnitude, phase)
- Visualize original FRD data
- Display interpolated data before IR reconstruction
- Convert to impulse response with adjustable parameters:
  - Sample rate selection (44.1 kHz to 192 kHz)
  - Delay compensation adjustment
- View reconstructed frequency response from IR
- Phase wrapping/unwrapping option
- Dark/Light theme support

## Building

```bash
cd /Users/arnauddemion/Projects/frd-to-ir
cargo build --release
```

## Running

```bash
cargo run --release
```

## Usage

1. Use **File → Open FRD...** to load an FRD file
2. Adjust the **Sample Rate** if needed (default: 96 kHz)
3. Adjust the **Delay** parameter to control IR timing
4. Toggle **Wrap Phase** to switch between wrapped and unwrapped phase display
5. View three graphs:
   - **FRD Data**: Original data points (solid) and interpolated curve (dashed)
   - **Impulse Response**: Time-domain representation
   - **Reconstructed Frequency Response**: Verification of the conversion

## FRD File Format

The application expects FRD files with three columns:
```
frequency_Hz  magnitude_dB  phase_degrees
```

Lines starting with `#`, `;`, or `*` are treated as comments.

## Technical Details

- Uses PCHIP (Piecewise Cubic Hermite Interpolation) for smooth interpolation
- Applies Hermitian symmetry for real-valued output
- Phase unwrapping to handle discontinuities
- Delay compensation for impulse positioning

## Dependencies

- eframe/egui: UI framework
- egui_plot: Plotting functionality
- rustfft: FFT operations
- num-complex: Complex number support
- interp1d: Additional interpolation support
- rfd: File dialogs
