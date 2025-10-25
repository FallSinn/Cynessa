param(
    [string]$ManifestPath = "../updates/file_manifest.json",
    [string]$Root = ".."
)

$manifest = Get-Content -Path $ManifestPath | ConvertFrom-Json
if (-not $manifest) {
    Write-Error "Unable to read manifest"
    exit 1
}

$errors = @()
foreach ($entry in $manifest.files) {
    $path = Join-Path -Path $Root -ChildPath $entry.path
    if (-not (Test-Path $path)) {
        $errors += "Missing: $($entry.path)"
        continue
    }
    $hash = (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
    if ($hash -ne $entry.sha256.ToLower()) {
        $errors += "Mismatch: $($entry.path)"
    }
}

if ($errors.Count -eq 0) {
    Write-Output "All files validated."
} else {
    $errors | ForEach-Object { Write-Output $_ }
    exit 1
}
