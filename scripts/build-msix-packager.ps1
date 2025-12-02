# Build MSIX package using cargo-packager
param(
    [string]$Target = "x86_64-pc-windows-msvc",
    [string]$OutputDir = "target/release",
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

Write-Host "Building MSIX package with cargo-packager..." -ForegroundColor Green
Write-Host "Target: $Target" -ForegroundColor Yellow
Write-Host "Version: $Version" -ForegroundColor Yellow

Set-Location $ProjectRoot

# Check if cargo-packager is installed
$PackagerCheck = & cargo packager --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "cargo-packager not found, installing..." -ForegroundColor Yellow
    & cargo install cargo-packager
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install cargo-packager"
    }
    Write-Host "cargo-packager installed successfully" -ForegroundColor Green
}

# Build the application if not skipped
if (-not $SkipBuild) {
    Write-Host "Building LinFIR application..." -ForegroundColor Blue
    & cargo build --release --features registration --target $Target
    if ($LASTEXITCODE -ne 0) {
        throw "Cargo build failed"
    }
    Write-Host "Build completed successfully" -ForegroundColor Green
}

# Create MSIX package using cargo-packager
Write-Host "Creating MSIX package..." -ForegroundColor Blue

$PackagerArgs = @(
    "packager",
    "--release",
    "--formats", "nsis,wix,msix",
    "--target", $Target
)

& cargo @PackagerArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "cargo-packager failed, trying with msix only..." -ForegroundColor Yellow

    # Try with just MSIX format
    $PackagerArgs = @(
        "packager",
        "--release",
        "--formats", "msix",
        "--target", $Target
    )

    & cargo @PackagerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create MSIX package with cargo-packager"
    }
}

# Find the generated MSIX file
$SearchPattern = "LinFIR*$Target*.msix"
$MsixFiles = Get-ChildItem -Path $OutputDir -Name $SearchPattern -Recurse

if (-not $MsixFiles) {
    # Try alternative search locations
    $AlternativeLocations = @(
        "target\$Target\release",
        "target\release\bundle",
        "target\packager"
    )

    foreach ($Location in $AlternativeLocations) {
        $FullPath = Join-Path $ProjectRoot $Location
        if (Test-Path $FullPath) {
            $MsixFiles = Get-ChildItem -Path $FullPath -Name $SearchPattern -Recurse
            if ($MsixFiles) {
                $OutputDir = $FullPath
                break
            }
        }
    }
}

if (-not $MsixFiles) {
    throw "No MSIX files found matching pattern: $SearchPattern"
}

$MsixFile = $MsixFiles[0]
$MsixPath = Join-Path $OutputDir $MsixFile

Write-Host "MSIX package created: $MsixPath" -ForegroundColor Green
Write-Host "Package size: $([math]::Round((Get-Item $MsixPath).Length / 1MB, 2)) MB" -ForegroundColor White

# Sign the package if certificate is provided
if (-not $SkipSigning -and $CertificatePath -and (Test-Path $CertificatePath)) {
    Write-Host "Signing MSIX package..." -ForegroundColor Blue

    # Find signtool.exe
    $WindowsKitsPaths = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.22621.0\x64",
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.19041.0\x64",
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.18362.0\x64",
        "${env:ProgramFiles}\Windows Kits\10\bin\10.0.22621.0\x64",
        "${env:ProgramFiles}\Windows Kits\10\bin\10.0.19041.0\x64"
    )

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

        $SignArgs += "`"$MsixPath`""

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

# Output final information
Write-Host "`nPackage Information:" -ForegroundColor Cyan
Write-Host "  File: $MsixPath" -ForegroundColor White
Write-Host "  Target: $Target" -ForegroundColor White
Write-Host "  Version: $Version" -ForegroundColor White
Write-Host "`nMSIX package build completed!" -ForegroundColor Green

# Test package validity (optional)
Write-Host "`nValidating package..." -ForegroundColor Blue
try {
    # Use PowerShell's built-in APPX validation
    $PackageInfo = Get-AppxPackageManifest -Package $MsixPath -ErrorAction Stop
    Write-Host "Package validation: PASSED" -ForegroundColor Green
    Write-Host "Package Identity: $($PackageInfo.Package.Identity.Name)" -ForegroundColor White
    Write-Host "Package Version: $($PackageInfo.Package.Identity.Version)" -ForegroundColor White
} catch {
    Write-Warning "Package validation failed: $($_.Exception.Message)"
    Write-Host "This might be normal if the package isn't installed" -ForegroundColor Yellow
}

# Return the package path for CI/CD
return $MsixPath
