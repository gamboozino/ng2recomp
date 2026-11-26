$ErrorActionPreference = "Stop"
$zipPath = "out/windows-deps/ffmpeg.zip"
$extractTemp = "out/windows-deps/ffmpeg_temp"
$finalDir = "out/windows-deps/ffmpeg"
$url = "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068-win64-lgpl-shared-9.0.zip"

New-Item -ItemType Directory -Force -Path "out/windows-deps" | Out-Null
Write-Host "Downloading FFmpeg from $url..."
curl.exe -fL -o $zipPath $url

if (Test-Path $zipPath) {
    Write-Host "Extracting FFmpeg package..."
    if (Test-Path $extractTemp) { Remove-Item -Recurse -Force $extractTemp }
    Expand-Archive -Path $zipPath -DestinationPath $extractTemp -Force
    $subFolder = Get-ChildItem -Path $extractTemp -Directory | Select-Object -First 1
    if (Test-Path $finalDir) { Remove-Item -Recurse -Force $finalDir }
    Move-Item -Path $subFolder.FullName -Destination $finalDir
    Remove-Item -Recurse -Force $extractTemp
    Remove-Item -Force $zipPath
    Write-Host "FFmpeg Windows Dev Package Ready at ${finalDir}:"
    Get-ChildItem $finalDir
} else {
    Write-Error "Download failed."
}
