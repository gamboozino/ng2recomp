$ErrorActionPreference = "Stop"

$llvmBin = "C:\Program Files\LLVM\bin"
if ($env:PATH -notlike "*$llvmBin*") {
    $env:PATH = "$llvmBin;$env:PATH"
}
$rexBin = Resolve-Path "$PSScriptRoot\..\thirdparty\rexglue-sdk\out\win-amd64"
if ((Test-Path $rexBin) -and ($env:PATH -notlike "*$rexBin*")) {
    $env:PATH = "$rexBin;$env:PATH"
}

Write-Host "Configuring ng2..."
cmake -B out/build/win-amd64-release `
    -DCMAKE_C_FLAGS="-march=x86-64-v2" `
    -DCMAKE_CXX_FLAGS="-march=x86-64-v2"

Write-Host "Building ng2..."
cmake --build out/build/win-amd64-release --target ng2 -j
