//! Build script for FRD to IR Converter
//!
//! This script handles platform-specific build configurations:
//! - On Windows: Embeds the application icon as a resource in the executable
//! - On other platforms: Does nothing

fn main() {
    // Only embed Windows resources when building for Windows
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();

        // Set the application icon
        res.set_icon("icons/icon.ico");

        // Set application metadata
        res.set("ProductName", "FRD to IR Converter");
        res.set("FileDescription", "Convert FRD frequency response data to impulse response");
        res.set("CompanyName", "DEM Audio");
        res.set(
            "LegalCopyright",
            "Copyright (c) 2025, DEM Audio, Arnaud Demion",
        );

        // Compile the resources
        if let Err(e) = res.compile() {
            eprintln!("Warning: Failed to compile Windows resources: {}", e);
            // Don't fail the build if resource compilation fails
        }
    }

    // For non-Windows platforms, we don't need to do anything
    #[cfg(not(target_os = "windows"))]
    {
        // Nothing to do on non-Windows platforms
    }
}
