echo "### Creating Local Nix Binary Cache ###"
PROJECT_ROOT_FOR_CACHE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # If run from a script in root
# Or if running commands manually:
# PROJECT_ROOT_FOR_CACHE="$PWD" 

NIX_LOCAL_CACHE_DIR="$PROJECT_ROOT_FOR_CACHE/nix_binary_cache_local"
SHELL_NIX_FILE="$PROJECT_ROOT_FOR_CACHE/shell.nix"

echo "Ensuring all dependencies for $SHELL_NIX_FILE are built..."
nix-build --no-out-link "$SHELL_NIX_FILE" # This populates /nix/store

echo "Instantiating $SHELL_NIX_FILE for cache creation..."
PROJECT_SHELL_DRV_PATH=$(nix-instantiate "$SHELL_NIX_FILE")
if [ -z "$PROJECT_SHELL_DRV_PATH" ]; then
    echo "ERROR: Failed to instantiate $SHELL_NIX_FILE. Cannot create cache."
    exit 1
fi
echo "Shell derivation path to cache: $PROJECT_SHELL_DRV_PATH"

echo "Preparing local Nix binary cache directory at: $NIX_LOCAL_CACHE_DIR"
rm -rf "$NIX_LOCAL_CACHE_DIR" # Clean slate
mkdir -p "$NIX_LOCAL_CACHE_DIR"

echo "Copying Nix store paths to local cache (this may take a while)..."
nix --extra-experimental-features "nix-command flakes" copy --to "file://$NIX_LOCAL_CACHE_DIR" "$PROJECT_SHELL_DRV_PATH"
if [ $? -ne 0 ]; then
    echo "ERROR: 'nix copy' failed. Local Nix binary cache may be incomplete."
    exit 1
fi

# Add to .gitignore if not already there
GITIGNORE_FILE_FOR_CACHE="$PROJECT_ROOT_FOR_CACHE/.gitignore"
CACHE_GITIGNORE_ENTRY_FOR_NIX="nix_binary_cache_local/"
if ! grep -qF -- "$CACHE_GITIGNORE_ENTRY_FOR_NIX" "$GITIGNORE_FILE_FOR_CACHE" 2>/dev/null; then
    echo "Adding '$CACHE_GITIGNORE_ENTRY_FOR_NIX' to $GITIGNORE_FILE_FOR_CACHE."
    echo "$CACHE_GITIGNORE_ENTRY_FOR_NIX" >> "$GITIGNORE_FILE_FOR_CACHE"
    sort -u "$GITIGNORE_FILE_FOR_CACHE" -o "$GITIGNORE_FILE_FOR_CACHE"
fi
echo "Local Nix binary cache created successfully at $NIX_LOCAL_CACHE_DIR"
echo "### Finished Creating Local Nix Binary Cache ###"