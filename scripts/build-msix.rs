use std::env;
use std::fs;
use std::path::Path;
use std::process::{exit, Command};

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command line arguments
    let target = args
        .get(1)
        .unwrap_or(&"x86_64-pc-windows-msvc".to_string())
        .clone();
    let skip_build = args.contains(&"--skip-build".to_string());
    let sign_package = args.contains(&"--sign".to_string());

    println!("🚀 Building MSIX package for LinFIR...");
    println!("📦 Target: {}", target);

    // Get project root directory
    let project_root = env::current_dir().expect("Failed to get current directory");

    // Read version from Cargo.toml
    let cargo_toml_path = project_root.join("Cargo.toml");
    let cargo_content = fs::read_to_string(&cargo_toml_path).expect("Failed to read Cargo.toml");

    let version =
        extract_version(&cargo_content).expect("Failed to extract version from Cargo.toml");

    println!("📋 Version: {}", version);

    // Build the application if not skipped
    if !skip_build {
        println!("🔨 Building LinFIR application...");
        let build_status = Command::new("cargo")
            .args(&[
                "build",
                "--release",
                "--features",
                "registration",
                "--target",
                &target,
            ])
            .current_dir(&project_root)
            .status()
            .expect("Failed to execute cargo build");

        if !build_status.success() {
            eprintln!("❌ Cargo build failed");
            exit(1);
        }
        println!("✅ Build completed successfully");
    }

    // Use cargo-packager to create MSIX
    println!("📦 Creating MSIX package with cargo-packager...");

    let mut packager_args = vec![
        "packager",
        "--release",
        "--formats",
        "msix",
        "--target",
        &target,
    ];

    let packager_status = Command::new("cargo")
        .args(&packager_args)
        .current_dir(&project_root)
        .status()
        .expect("Failed to execute cargo packager");

    if !packager_status.success() {
        eprintln!("❌ cargo-packager failed");
        eprintln!("💡 Make sure cargo-packager is installed: cargo install cargo-packager");
        exit(1);
    }

    // Find the generated MSIX file
    let output_dir = project_root.join("target").join("release");
    let msix_pattern = format!("LinFIR_{}_*.msix", version.replace(".", "_"));

    println!("✅ MSIX package created successfully!");

    // Sign the package if requested
    if sign_package {
        println!("🔐 Signing MSIX package...");

        let cert_path = env::var("MSIX_CERT_PATH")
            .expect("MSIX_CERT_PATH environment variable required for signing");
        let cert_password = env::var("MSIX_CERT_PASSWORD").ok();

        sign_msix_package(
            &output_dir,
            &msix_pattern,
            &cert_path,
            cert_password.as_deref(),
        );
    }

    println!("🎉 MSIX package build completed!");
    println!("📁 Check target/release/ for the generated .msix file");
}

fn extract_version(cargo_content: &str) -> Option<String> {
    // Parse version from Cargo.toml using regex-like approach
    for line in cargo_content.lines() {
        let line = line.trim();
        if line.starts_with("version") && line.contains("=") {
            if let Some(version_part) = line.split("=").nth(1) {
                let version = version_part.trim().trim_matches('"').trim_matches('\'');
                return Some(version.to_string());
            }
        }
    }
    None
}

fn sign_msix_package(
    output_dir: &Path,
    pattern: &str,
    cert_path: &str,
    cert_password: Option<&str>,
) {
    // Find signtool.exe in Windows SDK paths
    let windows_kits_paths = vec![
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64",
        r"C:\Program Files\Windows Kits\10\bin\10.0.22621.0\x64",
        r"C:\Program Files\Windows Kits\10\bin\10.0.19041.0\x64",
    ];

    let mut signtool_path = None;
    for kit_path in &windows_kits_paths {
        let test_path = Path::new(kit_path).join("signtool.exe");
        if test_path.exists() {
            signtool_path = Some(test_path);
            break;
        }
    }

    let signtool = signtool_path.expect("signtool.exe not found. Please install Windows 10 SDK.");

    // Find MSIX files matching pattern
    if let Ok(entries) = fs::read_dir(output_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let filename = entry.file_name();
                let filename_str = filename.to_string_lossy();

                if filename_str.ends_with(".msix") && filename_str.contains("LinFIR") {
                    println!("🔐 Signing: {}", filename_str);

                    let mut sign_args = vec![
                        "sign".to_string(),
                        "/fd".to_string(),
                        "SHA256".to_string(),
                        "/f".to_string(),
                        cert_path.to_string(),
                    ];

                    if let Some(password) = cert_password {
                        sign_args.push("/p".to_string());
                        sign_args.push(password.to_string());
                    }

                    sign_args.push(entry.path().to_string_lossy().to_string());

                    let sign_status = Command::new(&signtool)
                        .args(&sign_args)
                        .status()
                        .expect("Failed to execute signtool");

                    if sign_status.success() {
                        println!("✅ Package signed successfully");
                    } else {
                        eprintln!("⚠️  Failed to sign package");
                    }
                }
            }
        }
    }
}
