#!/bin/bash

# Import utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/logging.sh"
source "$SCRIPT_DIR/utils/os_detection.sh"
source "$SCRIPT_DIR/utils/install_config.sh"
source "$SCRIPT_DIR/utils/python_tools.sh"

# Version
VERSION="1.0.0"

# Total number of stages
TOTAL_STAGES=6
CURRENT_STAGE=0
STAGE_NAME=""

# Trap ctrl-c and call cleanup
trap cleanup INT

# Cleanup function to restore terminal settings
cleanup() {
    echo -e "\n${YELLOW}Setup interrupted.${NC}"
    exit 1
}

# Display title and logo
display_header() {
    local title="$1"
    
    echo ""
    echo -e "${CYAN}"
    echo ' ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗'
    echo ' ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝'
    echo ' ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  '
    echo ' ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  '
    echo ' ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗'
    echo ' ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝'
    echo -e "${NC}"
    echo -e "${BOLD}Second-Me Setup Script v${VERSION}${NC}"
    echo -e "${GRAY}$(date)${NC}\n"
    
    if [ -n "$title" ]; then
        echo -e "${CYAN}====== $title ======${NC}"
        echo ""
    fi
}

# Display stage start
display_stage() {
    local stage_num=$1
    local stage_name=$2
    CURRENT_STAGE=$stage_num
    STAGE_NAME=$stage_name
    
    echo ""
    echo -e "${CYAN}====== Stage $stage_num/$TOTAL_STAGES: $stage_name ======${NC}"
    echo ""
}

# Setup and configure package managers (npm)
check_npm() {
    log_step "Checking npm installation..."
    
    # Check if npm is already installed
    if ! command -v npm &>/dev/null; then
        log_error "npm not found - please install npm manually"
        
        # Get system identification and show installation recommendations
        local system_id=$(get_system_id)
        get_npm_recommendation "$system_id"
        
        return 1
    fi

    log_success "npm check passed"
    return 0
}

# Check Node.js installation
check_node() {
    log_step "Checking Node.js installation..."
    
    local node_cmd=""
    
    # Check for node command
    if command -v node &>/dev/null; then
        node_cmd="node"
    # Also check for nodejs command as it's used on some Linux distributions
    elif command -v nodejs &>/dev/null; then
        node_cmd="nodejs"
    else
        log_error "Node.js is not installed, please install Node.js manually"
        
        # Get system identification and show installation recommendations
        local system_id=$(get_system_id)
        get_node_recommendation "$system_id"
        
        return 1
    fi
    
    # Check version (if needed)
    local version=$($node_cmd --version 2>&1 | sed 's/v//')
    log_success "Node.js check passed, using $node_cmd version $version"
    return 0
}

# Check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# Helper function to install dependencies
install_python_dependency() {
    # Install Python packages using Poetry
    log_step "Installing Python packages using Poetry"
    
    # Check if pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        log_error "Missing pyproject.toml file"
        return 1
    fi
    
    # Install the checked-in dependency versions.
    log_info "Checking Poetry lockfile..."
    if ! poetry check --lock; then
        log_error "Missing or outdated lockfile; run poetry lock before setup"
        return 1
    fi
    
    # Install dependencies
    log_info "Using Poetry to install dependencies..."
    if ! poetry install --no-root --no-interaction; then
        log_error "Failed to install dependencies using Poetry"
        return 1
    fi
    
    # Verify key packages are installed using Poetry's own environment
    log_info "Verifying key packages using Poetry environment..."
    local required_packages=("flask" "chromadb" "langchain")
    for pkg in "${required_packages[@]}"; do
        if ! poetry run python -c "import $pkg" 2>/dev/null; then
            log_error "Package '$pkg' is not installed correctly in Poetry environment"
            return 1
        else
            log_info "Package '$pkg' is installed correctly in Poetry environment"
        fi
    done
    
    # Get and save the Poetry environment path
    local poetry_env_path=$(poetry env info -p 2>/dev/null)
    if [ -n "$poetry_env_path" ]; then
        log_info "Poetry virtual environment is located at: $poetry_env_path"
        # Create an activation script for convenience
        create_poetry_activate_script "$poetry_env_path"
    fi

    log_success "Python environment setup completed"
    log_info "------------------------------------------------------------------------------"
    log_info "To use this Python environment, you can:"
    log_info "1. Run 'poetry shell' to open a new shell with the virtual environment activated"
    log_info "2. Run 'source .poetry-venv/activate' to activate the environment in your current shell"
    log_info "3. Use 'poetry run python script.py' to run a single command without activating the environment"
    log_info "------------------------------------------------------------------------------"
    return 0
}

# Create a convenient activation script for the Poetry environment
create_poetry_activate_script() {
    local env_path="$1"
    local activate_dir=".poetry-venv"
    local activate_script="$activate_dir/activate"
    
    # Create directory if it doesn't exist
    mkdir -p "$activate_dir"
    
    # Create activation script
    cat > "$activate_script" << EOF
#!/bin/bash
# Activation script for Poetry virtual environment

# Source the actual virtual environment activate script
source "$env_path/bin/activate"

# Print confirmation message
echo "Poetry virtual environment activated: $env_path"
echo "Use 'deactivate' command to exit this environment"
EOF
    
    # Make script executable
    chmod +x "$activate_script"
    log_info "Created Poetry environment activation script at: $activate_script"
}

install_graphrag() {
    log_step "Installing graphrag"
    
    # Check the current graphrag version in Poetry environment
    log_step "Checking graphrag version in Poetry environment"
    GRAPHRAG_VERSION=$(poetry run pip show graphrag 2>/dev/null | grep "Version:" | cut -d " " -f2)
    GRAPHRAG_TARGET="1.2.1.dev27"
    GRAPHRAG_LOCAL_PATH="dependencies/graphrag-${GRAPHRAG_TARGET}.tar.gz"

    if [ ! -f "$GRAPHRAG_LOCAL_PATH" ]; then
        log_error "Local GraphRAG package not found: $GRAPHRAG_LOCAL_PATH"
        return 1
    fi
    local package_hash=$(poetry run python -c 'import hashlib,sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").hexdigest())' "$GRAPHRAG_LOCAL_PATH")
    local package_marker="$(poetry run python -c 'import sys; print(sys.prefix)')/.secondme-graphrag.sha256"
    # This ZIP uses dynamic versioning; its filename is not the installed version.
    if [[ -z "$GRAPHRAG_VERSION" || ! -f "$package_marker" || "$(cat "$package_marker" 2>/dev/null)" != "$package_hash" ]]; then
        log_info "Installing patched GraphRAG archive in Poetry environment..."
        if [ -f "$GRAPHRAG_LOCAL_PATH" ]; then
            log_info "Installing graphrag from local file using Poetry..."
            local install_path="$GRAPHRAG_LOCAL_PATH"
            local package_tmp=""
            if poetry run python -c 'import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$GRAPHRAG_LOCAL_PATH"; then
                package_tmp=$(mktemp -d)
                install_path="$package_tmp/graphrag-${GRAPHRAG_TARGET}.zip"
                cp "$GRAPHRAG_LOCAL_PATH" "$install_path"
            fi
            if ! poetry run pip install --no-deps --force-reinstall "$install_path"; then
                [[ -z "$package_tmp" ]] || rm -rf "$package_tmp"
                log_error "Failed to install graphrag from local file"
                return 1
            fi
        else
            log_error "Local graphrag package not found at: $GRAPHRAG_LOCAL_PATH"
            log_error "Please ensure the graphrag package exists in the dependencies directory"
            return 1
        fi
        [[ -z "$package_tmp" ]] || rm -rf "$package_tmp"
        local installed_version=$(poetry run python -c 'from importlib.metadata import version; print(version("graphrag"))')
        [[ -n "$installed_version" ]] || return 1
        printf '%s\n' "$package_hash" > "$package_marker"
        log_success "Patched GraphRAG installed successfully (metadata version: $installed_version)"
    else
        log_success "Patched GraphRAG archive hash matches, skipping installation"
    fi
    
    return 0
}

# Build llama.cpp
build_llama() {
    log_section "BUILDING LLAMA.CPP"
    
    local backend=${LLAMA_BACKEND:-}
    if [[ -z "$backend" ]]; then
        backend=cpu
        if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
            backend=metal
        elif command -v nvcc &>/dev/null; then
            backend=cuda
        elif command -v glslc &>/dev/null && command -v vulkaninfo &>/dev/null && vulkaninfo --summary &>/dev/null; then
            backend=vulkan
        fi
    fi
    log_info "llama.cpp backend: $backend (override with LLAMA_BACKEND=cpu|metal|cuda|vulkan)"
    bash scripts/build_llama.sh "$backend"

}

# Set up frontend environment
build_frontend() {
    log_section "SETTING UP FRONTEND"
    
    FRONTEND_DIR="lpm_frontend"
    
    # Enter frontend directory
    cd "$FRONTEND_DIR" || {
        log_error "Failed to enter frontend directory: $FRONTEND_DIR"
        log_error "Please ensure the directory exists and you have permission to access it."
        return 1
    }
    
    # Check if dependencies have been installed
    if [ -d "node_modules" ]; then
        log_info "Found existing node_modules, checking for updates..."
        if [ -f "package-lock.json" ]; then
            log_info "Using existing package-lock.json..."
            # Run npm install even if package-lock.json exists to ensure dependencies are complete
            log_info "Running npm install to ensure dependencies are complete..."
            if ! npm install; then
                log_error "Failed to install frontend dependencies with existing package-lock.json"
                log_error "Try removing node_modules directory and package-lock.json, then run setup again"
                cd ..
                return 1
            fi
        else
            log_info "Installing dependencies..."
            if ! npm install; then
                log_error "Failed to install frontend dependencies"
                log_error "Check your npm configuration and network connection"
                log_error "You can try running 'npm install' manually in the $FRONTEND_DIR directory"
                cd ..
                return 1
            fi
        fi
    else
        log_info "Installing dependencies..."
        if ! npm install; then
            log_error "Failed to install frontend dependencies"
            log_error "Check your npm configuration and network connection"
            log_error "You can try running 'npm install' manually in the $FRONTEND_DIR directory"
            cd ..
            return 1
        fi
    fi
    
    # Verify that the installation was successful
    if [ ! -d "node_modules" ]; then
        log_error "node_modules directory not found after npm install"
        log_error "Frontend dependencies installation failed"
        cd ..
        return 1
    fi
    
    log_success "Frontend dependencies installed successfully"
    cd ..
    log_section "FRONTEND SETUP COMPLETE"
}

# Show help information
show_help() {
    echo -e "${BOLD}Second-Me Setup Script v${VERSION}${NC}"
    echo -e "Usage: $0 [options] [command]"
    echo
    echo -e "Commands:"
    echo -e "  python\t\tSetup Python environment only"
    echo -e "  llama\t\t\tBuild llama.cpp only"
    echo -e "  frontend\t\tSetup frontend project only"
    echo -e "  (no command)\t\tPerform full installation"
    echo
    echo -e "Options:"
    echo -e "  --help\t\tShow this help information"
    echo -e "  --require-confirmation\tRequire confirmation when warnings are present"
    echo
    echo -e "Examples:"
    echo -e "  $0 \t\t\tPerform full installation"
    echo -e "  $0 python\t\tSetup Python environment only"
    echo -e "  $0 --require-confirmation\tRequire confirmation when warnings are present"
    echo
    echo -e "For a complete list of all available commands, run:"
    echo -e "  make help"
}

# Check system requirements
check_system_requirements() {
    log_step "Checking system requirements"
    
    # Detect system type
    local system_type=$(uname -s)
    log_info "Detected system type: $system_type"
    
    # Only check macOS version if on Mac
    if [[ "$system_type" == "Darwin" ]]; then
        local macos_version=$(sw_vers -productVersion)
        log_info "Detected macOS version: $macos_version"
        
        local major_version=$(echo "$macos_version" | cut -d. -f1)
        if [[ "$major_version" -lt 14 ]]; then
            log_error "This script requires macOS 14 (Sonoma) or later. Your version: $macos_version"
            return 1
        fi
    fi

    log_success "System requirements check passed"
    return 0
}

# Check required configuration files
check_config_files() {
    log_step "Checking necessary configuration files"
    
    # Check for .env file
    if [[ ! -f ".env" ]]; then
        log_error "Missing .env file"
        return 1
    fi
    
    log_success "All necessary configuration files are present"
    return 0
}

# Check directory permissions
check_directory_permissions() {
    log_step "Checking directory permissions"
    local errors=0
    local directories=("." "./scripts" "./run" "./logs")
    
    for dir in "${directories[@]}"; do
        if [[ ! -w "$dir" ]]; then
            log_error "Directory without write permission: $dir"
            errors=$((errors + 1))
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log_success "Directory permissions check passed"
        return 0
    else
        return 1
    fi
}

# Check for potential conflicts
check_potential_conflicts() {
    log_info "Checking for potential conflicts"

    # System requirements check
    if ! check_system_requirements; then
        log_error "System requirements check failed"
        exit 1
    fi
    
    # Configuration files check
    if ! check_config_files; then
        log_error "Configuration files check failed"
        exit 1
    fi
    
    # Directory permissions check
    if ! check_directory_permissions; then
        log_error "Directory permissions check failed"
        exit 1
    fi

    if ! check_python; then
        log_error "python check failed, please install python first"
        exit 1
    fi
    
    if ! check_node; then
        log_error "Node.js check failed"
        exit 1
    fi
    
    if ! check_npm; then
        log_error "npm check failed"
        exit 1
    fi
    
    if ! check_cmake; then
        log_error "cmake check and installation failed"
        exit 1
    fi

    if ! check_poetry; then
        log_error "poetry check failed, please install poetry first"
        exit 1
    fi
    
    return 0
}

check_python() {
    log_step "Checking for python installation"
    
    # Get the appropriate Python command
    local python_cmd=$(get_python_command)
    
    if [ -z "$python_cmd" ]; then
        log_error "python is not installed, please install python manually"
        
        # Get system identification and show installation recommendations
        local system_id=$(get_system_id)
        get_python_recommendation "$system_id"
        
        return 1
    fi
    
    # version > 3.12
    local version=$($python_cmd --version 2>&1 | cut -d ' ' -f 2)
    if [[ "$version" < "3.12" ]]; then
        log_error "python version $version is not supported, please install python 3.12 or higher"
        return 1
    fi
    
    log_success "python check passed, using $python_cmd version $version"
    return 0
}

check_poetry() {
    log_step "Checking for poetry installation"
    
    if ! command -v poetry &>/dev/null; then
        log_error "poetry is not installed, please install poetry manually"
        
        # Get system identification and show installation recommendations
        local system_id=$(get_system_id)
        get_poetry_recommendation "$system_id"
        
        return 1
    fi
    
    log_success "poetry check passed"
    return 0
}

# Check and install cmake if not present
check_cmake() {
    log_step "Checking for cmake installation"
    
    if ! command -v cmake &>/dev/null; then
        log_warning "cmake is not installed, please install cmake manually"
        
        # Get system identification and show installation recommendations
        local system_id=$(get_system_id)
        get_cmake_recommendation "$system_id"
        
        return 1
    fi
    
    log_success "cmake check passed"
    return 0
}

# Check if SQLite is installed and available
check_sqlite() {
    log_step "Checking SQLite"
    
    if ! check_command "sqlite3"; then
        log_warning "SQLite3 is not installed or not in your PATH"
        
        log_error "Please install SQLite before continuing, database operations require this dependency"

        # Get system identification and show installation recommendations
        local system_id=$(get_system_id)
        get_sqlite_recommendation "$system_id"
        
        return 1
    fi
    
    # SQLite is installed
    local version=$(sqlite3 --version | awk '{print $1}')
    log_success "SQLite check passed, version $version"
    return 0
}

# Parse command line arguments
parse_args() {
    REQUIRE_CONFIRMATION=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --require-confirmation)
                REQUIRE_CONFIRMATION=true
                shift
                ;;
            python|llama|frontend)
                COMPONENT="$1"
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}


# Main function
main() {
    # Display welcome message
    display_header "Second-Me Complete Installation"
    
    # Parse command line arguments
    parse_args "$@"
    
    # All pre-installation checks
    log_section "Running pre-installation checks"
    
    # 1. Basic tools check (most fundamental)
    if ! check_potential_conflicts; then
        log_error "Basic tools check failed"
        exit 1
    fi
    
    # Check SQLite installation
    if ! check_sqlite; then
        log_error "SQLite check failed"
        exit 1
    fi
    
    # Start installation process
    log_section "Starting installation"
    
    if ! install_python_dependency; then
        log_error "Failed to install python dependencies"
        exit 1
    fi

    if ! install_graphrag; then
        log_error "Failed to install graphrag"
        exit 1
    fi

    # 3. Build llama.cpp
    if ! build_llama; then
        exit 1
    fi
    
    # 4. Build frontend
    if ! build_frontend; then
        exit 1
    fi

    log_success "Installation complete!"
    return 0
}

# Start execution
main "$@"
