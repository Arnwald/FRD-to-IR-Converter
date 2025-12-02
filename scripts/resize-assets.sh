#!/bin/bash

# Simple script to resize MSIX assets to correct dimensions for Microsoft Store
# Uses sips (macOS built-in tool)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
SOURCE_ICON="$PROJECT_ROOT/icons/icon.png"
ASSETS_DIR="$PROJECT_ROOT/msix/Assets"

echo "🖼️  Resizing MSIX assets..."
echo "Source: $SOURCE_ICON"
echo "Target: $ASSETS_DIR"
echo ""

# Check if source exists
if [ ! -f "$SOURCE_ICON" ]; then
    echo "❌ Source icon not found: $SOURCE_ICON"
    exit 1
fi

# Create assets directory
mkdir -p "$ASSETS_DIR"

# Function to resize with sips
resize_asset() {
    local name="$1"
    local width="$2"
    local height="$3"
    local output="$ASSETS_DIR/$name"

    echo "Resizing $name to ${width}x${height}..."
    sips -z "$height" "$width" "$SOURCE_ICON" --out "$output" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ $name"
    else
        echo "❌ Failed to resize $name"
        return 1
    fi
}

# Resize each asset (only mandatory ones to avoid logo deformation)
resize_asset "StoreLogo.png" 50 50
resize_asset "Square44x44Logo.png" 44 44
resize_asset "Square150x150Logo.png" 150 150

echo ""
echo "🎉 Asset resizing completed!"
echo ""
echo "Verification:"
for asset in "$ASSETS_DIR"/*.png; do
    if [ -f "$asset" ]; then
        size=$(sips -g pixelWidth -g pixelHeight "$asset" 2>/dev/null | grep -E "pixelWidth|pixelHeight" | awk '{print $2}' | tr '\n' 'x' | sed 's/x$//')
        echo "  $(basename "$asset"): ${size}"
    fi
done

echo ""
echo "Next steps:"
echo "1. Test the MSIX build with: scripts/build-msix.ps1"
echo "2. Check assets look correct in: msix/Assets/"
