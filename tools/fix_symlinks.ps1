$mspackDir = "thirdparty/rexglue-sdk/thirdparty/libmspack"
$files = Get-ChildItem -Path $mspackDir -Recurse -File

foreach ($f in $files) {
    if ($f.Length -lt 100 -and ($f.Name.EndsWith(".c") -or $f.Name.EndsWith(".h"))) {
        $content = Get-Content -Path $f.FullName -Raw
        if ($content.Contains("/") -and $content.Trim().EndsWith($f.Name)) {
            $targetRel = $content.Trim()
            $targetFull = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($f.DirectoryName, $targetRel))
            if (Test-Path $targetFull) {
                Write-Host "Fixing symlink: $($f.FullName) -> $targetFull"
                Copy-Item -Path $targetFull -Destination $f.FullName -Force
            }
        }
    }
}
Write-Host "Symlink check done."
