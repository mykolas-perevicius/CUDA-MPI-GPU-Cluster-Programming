#!/bin/bash

echo "Restructuring repository for CS485..."

# --- Configuration ---
ASSIGNMENTS_DIR="assignments" # Your current assignments directory
HOMEWORKS_DIR="homeworks"   # Target directory name
SCRIPTS_DIR="scripts"
CONFIG_DIR="config"
TEMPLATES_DIR="templates"

# --- Safety Checks ---
if [ ! -d "$ASSIGNMENTS_DIR" ] && [ ! -d "$HOMEWORKS_DIR" ]; then
    echo "Neither '$ASSIGNMENTS_DIR' nor '$HOMEWORKS_DIR' directory found. Assuming fresh setup."
    # Create standard directories if none exist
    mkdir -p "$HOMEWORKS_DIR"
    mkdir -p "$SCRIPTS_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$TEMPLATES_DIR"
else
    # --- Rename assignments to homeworks ---
    if [ -d "$ASSIGNMENTS_DIR" ]; then
        echo "Renaming '$ASSIGNMENTS_DIR' to '$HOMEWORKS_DIR'..."
        mv "$ASSIGNMENTS_DIR" "$HOMEWORKS_DIR" || { echo "Failed to rename assignments dir."; exit 1; }
    elif [ ! -d "$HOMEWORKS_DIR" ]; then
         mkdir -p "$HOMEWORKS_DIR"
         echo "Created '$HOMEWORKS_DIR' directory."
    fi
fi

# --- Create/Ensure Core Directories ---
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$TEMPLATES_DIR"
touch "$TEMPLATES_DIR/.gitkeep" # Keep templates dir even if empty initially

# --- Clean up top-level build artifacts if they exist ---
echo "Cleaning up potential top-level build artifacts..."
rm -rf ./build/
rm -f ./compile_commands.json
rm -f ./CMakeCache.txt
# Note: We will add 'build/' directories *inside* each homework later via scaffolding

# --- Reorganize Scripts ---
echo "Reorganizing scripts into '$SCRIPTS_DIR'..."
if [ -f "./generate_homework.sh" ]; then
    mv ./generate_homework.sh "$SCRIPTS_DIR/scaffold_hw.sh" || echo "Warning: Could not move generate_homework.sh"
fi
if [ -f "./scripts/master_test.sh" ]; then
    mv ./scripts/master_test.sh "$SCRIPTS_DIR/run_hw.sh" || echo "Warning: Could not move master_test.sh"
fi
if [ -f "./scripts/connectivity_test.sh" ]; then
    mv ./scripts/connectivity_test.sh "$SCRIPTS_DIR/check_cluster.sh" || echo "Warning: Could not move connectivity_test.sh"
fi
# Add placeholders for missing essential scripts
touch "$SCRIPTS_DIR/test_hw.sh"
touch "$SCRIPTS_DIR/package_hw.sh"
chmod +x "$SCRIPTS_DIR"/*.sh # Make scripts executable

# --- Handle CMakeLists.txt and package.sh within homeworks ---
echo "Checking for misplaced build/package files within homework directories..."
if [ -d "$HOMEWORKS_DIR" ]; then
    find "$HOMEWORKS_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r hw_dir; do
        # Move CMakeLists.txt if it's outside src/ (adjust if needed)
        if [ -f "$hw_dir/CMakeLists.txt" ]; then
             echo "Note: Found CMakeLists.txt in $hw_dir. Ensure it's correctly configured."
             # Decide if you want to move it, template likely better
             # mv "$hw_dir/CMakeLists.txt" "$hw_dir/CMakeLists.txt.old"
        fi
        # Remove per-homework package.sh scripts
        if [ -f "$hw_dir/package.sh" ]; then
            echo "Removing per-homework package script: $hw_dir/package.sh"
            rm "$hw_dir/package.sh"
        fi
        # Ensure src directory exists
        mkdir -p "$hw_dir/src"
    done
fi


# --- Remove potentially obsolete top-level files/dirs ---
if [ -d "./cmake" ]; then
    echo "Removing obsolete './cmake' directory..."
    rm -rf "./cmake"
fi
if [ -f "./setup_github_ci.sh" ]; then
    echo "Removing obsolete './setup_github_ci.sh'..."
    rm -f "./setup_github_ci.sh"
fi

# --- Setup Config Template ---
CONFIG_TEMPLATE="$CONFIG_DIR/cluster.conf.template"
if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "Creating '$CONFIG_TEMPLATE'..."
    # Create the template file here (content provided separately)
    echo "# Cluster Configuration Template" > "$CONFIG_TEMPLATE"
    echo "# Copy to config/cluster.conf and fill in your details" >> "$CONFIG_TEMPLATE"
    echo "" >> "$CONFIG_TEMPLATE"
    echo "MANAGER_HOST=node0" >> "$CONFIG_TEMPLATE"
    echo "MANAGER_IP=192.168.1.100 # Example IP" >> "$CONFIG_TEMPLATE"
    echo "WORKER_HOSTS=(\"node1\")" >> "$CONFIG_TEMPLATE"
    echo "WORKER_IPS=(\"192.168.1.101\") # Example IP" >> "$CONFIG_TEMPLATE"
    echo "CLUSTER_USER=$(whoami) # Default to current user" >> "$CONFIG_TEMPLATE"
    echo "REMOTE_CODE_DIR=/tmp # Example path for syncing code if needed" >> "$CONFIG_TEMPLATE"
fi

# --- Update .gitignore ---
GITIGNORE=".gitignore"
echo "Updating '$GITIGNORE'..."
# Add entries if they don't exist
grep -qxF 'build/' "$GITIGNORE" || echo 'build/' >> "$GITIGNORE"
grep -qxF 'config/cluster.conf' "$GITIGNORE" || echo 'config/cluster.conf' >> "$GITIGNORE"
grep -qxF '*.tgz' "$GITIGNORE" || echo '*.tgz' >> "$GITIGNORE"
grep -qxF '*.o' "$GITIGNORE" || echo '*.o' >> "$GITIGNORE"
grep -qxF '*.exe' "$GITIGNORE" || echo '*.exe' >> "$GITIGNORE" # If applicable
grep -qxF 'template' "$GITIGNORE" || echo 'template' >> "$GITIGNORE" # The executable name

echo "Restructuring complete. Please review the changes."
echo "Make sure to populate the scripts in '$SCRIPTS_DIR' and templates in '$TEMPLATES_DIR' with the correct content."