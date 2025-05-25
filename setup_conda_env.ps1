Write-Host "`n📦 NLP Project Environment Setup"
Write-Host "--------------------------------`n"

# Ask for target directory
$envDir = Read-Host "Enter the full path where you want to set up the environment (`leave blank for current directory`)"

# Use current directory if blank
if ([string]::IsNullOrWhiteSpace($envDir)) {
    $envDir = Get-Location
    Write-Host "📂 No directory specified. Using current directory: $envDir"
} else {
    Write-Host "📂 Using specified directory: $envDir"
}

# Ensure the directory exists
if (-Not (Test-Path $envDir)) {
    Write-Host "❌ Directory does not exist: $envDir"
    exit 1
}

# Copy environment.yml
Copy-Item -Path "environment.yml" -Destination "$envDir\environment.yml" -Force

# Change to that directory
Set-Location $envDir

# Run conda to create the environment
Write-Host "`n⚙️  Creating Conda environment from environment.yml..."
conda env create -f environment.yml

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Environment created successfully!"
    Write-Host "💡 To activate it, run:"
    Write-Host "   conda activate nlp-env"
} else {
    Write-Host "`n❌ Failed to create the environment. Please check environment.yml and Conda installation."
}
