# Chay nhan dien mu bao hiem tren video bang GPU.
#
# Cach dung (mo PowerShell tai thu muc IOT-main):
#   .\run_video.ps1                                  # chay video demo (dung tu anh VALID)
#   .\run_video.ps1 -Source "videos\cua_toi.mp4"     # chay video cua ban (dat trong thu muc IOT)
#   .\run_video.ps1 -MaxFrames 300                   # chay thu 300 frame dau
#   .\run_video.ps1 -NoTrack                         # chi nhan dien tung frame, khong gan ID
#   .\run_video.ps1 -Conf 0.4                        # doi nguong tin cay

param(
    [string]$Source = "results/predictions/video_sources/demo_val.mp4",
    [double]$Conf = 0,
    [int]$ImgSz = 416,
    [int]$MaxFrames = 0,
    [switch]$NoTrack
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Source)) {
    if ($Source -eq "results/predictions/video_sources/demo_val.mp4") {
        Write-Host "Chua co video demo, dang tao tu anh VALID..."
        & $python scripts/make_demo_video.py --split val --shuffle-seed 42
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Error "Khong tim thay video: $Source"
    }
}

$cmdArgs = @("scripts/predict_video.py", "--source", $Source, "--imgsz", $ImgSz, "--device", "cuda")
if ($Conf -gt 0) { $cmdArgs += @("--conf", $Conf) }
if ($MaxFrames -gt 0) { $cmdArgs += @("--max-frames", $MaxFrames) }
if ($NoTrack) { $cmdArgs += "--no-track" }

& $python @cmdArgs
exit $LASTEXITCODE
