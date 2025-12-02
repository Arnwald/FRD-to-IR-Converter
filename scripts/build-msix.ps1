# Build MSIX package for Windows Store distribution
param(
    [string]$Target = "x86_64-pc-windows-msvc",
    [string]$OutputDir = "target/msix",
    [string]$CertificatePath = "",
    [string]$CertificatePassword = "",
    [switch]$SkipBuild = $false,
    [switch]$SkipSigning = $false
)

$ErrorActionPreference = "Stop"

# Read version from Cargo.toml
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CargoTomlPath = Join-Path $ProjectRoot "Cargo.toml"
if (-not (Test-Path $CargoTomlPath)) {
    throw "Cargo.toml not found at: $CargoTomlPath"
}

$CargoContent = Get-Content $CargoTomlPath -Raw
if ($CargoContent -match 'version\s*=\s*"([^"]+)"') {
    $Version = $Matches[1]
    Write-Host "Version read from Cargo.toml: $Version" -ForegroundColor Green
} else {
    throw "Could not extract version from Cargo.toml"
}

Write-Host "Building MSIX package for LinFIR..." -ForegroundColor Green
Write-Host "Target: $Target" -ForegroundColor Yellow
Write-Host "Version: $Version" -ForegroundColor Yellow

# Define paths - select manifest based on target architecture
$ManifestTemplate = if ($Target -eq "x86_64-pc-windows-msvc") {
    Join-Path $ProjectRoot "msix\Package-x86_64.appxmanifest"
} elseif ($Target -eq "aarch64-pc-windows-msvc") {
    Join-Path $ProjectRoot "msix\Package-aarch64.appxmanifest"
} else {
    Join-Path $ProjectRoot "msix\Package.appxmanifest"
}
# Default binary path (will be updated for cross-compilation)
$BinaryPath = Join-Path $ProjectRoot "target\release\LinFIR.exe"
$PackageDir = Join-Path $ProjectRoot "$OutputDir\package"
$ManifestPath = Join-Path $PackageDir "AppxManifest.xml"
# Create architecture-specific package name
$ArchName = if ($Target -eq "x86_64-pc-windows-msvc") { "x64" } elseif ($Target -eq "aarch64-pc-windows-msvc") { "ARM64" } else { "neutral" }
$OutputPackage = Join-Path $ProjectRoot "$OutputDir\LinFIR-$ArchName-$Version.msix"

# Create output directories
Write-Host "Creating package directory structure..." -ForegroundColor Blue
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageDir "Assets") -Force | Out-Null

# Update binary path for cross-compilation
if ($Target -ne "x86_64-pc-windows-msvc") {
    $BinaryPath = Join-Path $ProjectRoot "target\$Target\release\LinFIR.exe"
}

# Build the application if not skipped
if (-not $SkipBuild) {
    Write-Host "Building LinFIR application..." -ForegroundColor Blue
    Set-Location $ProjectRoot
    & cargo build --release --features registration --target $Target
    if ($LASTEXITCODE -ne 0) {
        throw "Cargo build failed"
    }
}

# Verify binary exists
if (-not (Test-Path $BinaryPath)) {
    throw "Binary not found at: $BinaryPath"
}

# Copy executable
Write-Host "Copying application executable..." -ForegroundColor Blue
Copy-Item $BinaryPath $PackageDir -Force

# Copy properly sized PNG assets for Microsoft Store
Write-Host "Copying MSIX assets..." -ForegroundColor Blue

# Copy the pre-generated MSIX assets
$MsixAssetsDir = Join-Path $ProjectRoot "msix\Assets"
if (Test-Path $MsixAssetsDir) {
    Write-Host "Using pre-generated MSIX assets..." -ForegroundColor Green
    Copy-Item "$MsixAssetsDir\*" (Join-Path $PackageDir "Assets") -Force
} else {
    Write-Warning "No MSIX assets found. Run resize-assets.sh to generate them first."
}

# Copy additional resources if they exist
$ResourcesDir = Join-Path $ProjectRoot "assets"
if (Test-Path $ResourcesDir) {
    Write-Host "Copying additional resources..." -ForegroundColor Blue
    Copy-Item "$ResourcesDir\*" $PackageDir -Recurse -Force
}

# Update manifest with current version
Write-Host "Updating package manifest..." -ForegroundColor Blue
Write-Host "Using manifest: $ManifestTemplate" -ForegroundColor Yellow
$ManifestMinimal = Join-Path $ProjectRoot "msix\Package-minimal.appxmanifest"

if (Test-Path $ManifestTemplate) {
    $ManifestContent = Get-Content $ManifestTemplate -Raw
    $ManifestContent = $ManifestContent -replace "{{VERSION}}", "$Version.0"
    $ManifestContent | Set-Content $ManifestPath -Encoding UTF8
} elseif (Test-Path $ManifestMinimal) {
    Write-Host "Using minimal manifest as fallback..." -ForegroundColor Yellow
    $ManifestContent = Get-Content $ManifestMinimal -Raw
    $ManifestContent = $ManifestContent -replace "{{VERSION}}", "$Version.0"
    $ManifestContent | Set-Content $ManifestPath -Encoding UTF8
} else {
    throw "No manifest template found at: $ManifestTemplate or $ManifestMinimal"
}

# Find makeappx.exe in Windows SDK
$WindowsKitsPaths = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.22621.0\x64",
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.19041.0\x64",
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.18362.0\x64",
    "${env:ProgramFiles}\Windows Kits\10\bin\10.0.22621.0\x64",
    "${env:ProgramFiles}\Windows Kits\10\bin\10.0.19041.0\x64"
)

$MakeAppX = $null
foreach ($Path in $WindowsKitsPaths) {
    $TestPath = Join-Path $Path "makeappx.exe"
    if (Test-Path $TestPath) {
        $MakeAppX = $TestPath
        break
    }
}

if (-not $MakeAppX) {
    throw "makeappx.exe not found. Please install Windows 10 SDK."
}

Write-Host "Using makeappx: $MakeAppX" -ForegroundColor Yellow

# Create the MSIX package
Write-Host "Creating MSIX package..." -ForegroundColor Blue
& $MakeAppX pack /d $PackageDir /p $OutputPackage /overwrite
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed with main manifest, trying minimal manifest..." -ForegroundColor Yellow

    # Try with minimal manifest
    if (Test-Path $ManifestMinimal) {
        $ManifestContent = Get-Content $ManifestMinimal -Raw
        $ManifestContent = $ManifestContent -replace "{{VERSION}}", "$Version.0"
        $ManifestContent | Set-Content $ManifestPath -Encoding UTF8

        Write-Host "Retrying MSIX package creation with minimal manifest..." -ForegroundColor Blue
        & $MakeAppX pack /d $PackageDir /p $OutputPackage /overwrite
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create MSIX package even with minimal manifest"
        }
        Write-Host "Successfully created MSIX package with minimal manifest" -ForegroundColor Green
    } else {
        throw "Failed to create MSIX package and no minimal manifest available"
    }
}

Write-Host "MSIX package created: $OutputPackage" -ForegroundColor Green
Write-Host "Architecture: $ArchName" -ForegroundColor Green

# Sign the package if certificate is provided
if (-not $SkipSigning -and $CertificatePath -and (Test-Path $CertificatePath)) {
    Write-Host "Signing MSIX package..." -ForegroundColor Blue

    # Find signtool.exe
    $SignTool = $null
    foreach ($Path in $WindowsKitsPaths) {
        $TestPath = Join-Path $Path "signtool.exe"
        if (Test-Path $TestPath) {
            $SignTool = $TestPath
            break
        }
    }

    if (-not $SignTool) {
        Write-Warning "signtool.exe not found. Package will not be signed."
    } else {
        $SignArgs = @(
            "sign",
            "/fd", "SHA256",
            "/f", "`"$CertificatePath`""
        )

        if ($CertificatePassword) {
            $SignArgs += @("/p", $CertificatePassword)
        }

        $SignArgs += "`"$OutputPackage`""

        & $SignTool @SignArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Package signed successfully" -ForegroundColor Green
        } else {
            Write-Warning "Failed to sign package"
        }
    }
} else {
    Write-Host "Skipping package signing" -ForegroundColor Yellow
}

# Output package information
Write-Host "Package Information:" -ForegroundColor Cyan
Write-Host "  File: $OutputPackage" -ForegroundColor White
Write-Host "  Architecture: $ArchName" -ForegroundColor White
Write-Host "  Target: $Target" -ForegroundColor White
Write-Host "  Size: $([math]::Round((Get-Item $OutputPackage).Length / 1MB, 2)) MB" -ForegroundColor White
Write-Host "`nMSIX package build completed!" -ForegroundColor Green

# Return the package path for CI/CD
return $OutputPackage
