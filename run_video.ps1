# Chay nhan dien mu bao hiem tren video (kem ID doc lap, mau Xanh: With Helmet, mau Do: Without Helmet).
#
# Cach dung (mo PowerShell tai thu muc IOT):
#   .\run_video.ps1                                  # Chay video demo va luu ket qua
#   .\run_video.ps1 -View                            # VUA CHAY VUA XEM VIDEO TRUC TIEP
#   .\run_video.ps1 -Open                            # Tu dong mo video sau khi xu ly xong
#   .\run_video.ps1 -View -Open                      # Vua xem truc tiep vua mo lai sau khi xong
#   .\run_video.ps1 -Webcam -View                    # Mo Webcam may tinh de xem truc tiep
#   .\run_video.ps1 -Source "videos\cua_toi.mp4"     # Chay video tuy chon
#   .\run_video.ps1 -Weights "weights\best.pt"       # Su dung model best.pt vua train tren Colab
#   .\run_video.ps1 -MaxFrames 300                   # Chay 300 frame dau
#   .\run_video.ps1 -Conf 0.4                        # Nguong tin cay

param(
    [string]$Source = "results/predictions/video_sources/demo_val.mp4",
    [string]$Weights = "",
    [double]$Conf = 0,
    [int]$ImgSz = 416,
    [int]$MaxFrames = 0,
    [switch]$NoTrack,
    [switch]$View,
    [switch]$Open,
    [switch]$Webcam,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = ".\.venv\Scripts\python.exe"

if ($Webcam) {
    $Source = "0"
    $View = $true
} elseif (-not (Test-Path $Source)) {
    if ($Source -eq "results/predictions/video_sources/demo_val.mp4") {
        Write-Host "Chua co video demo, dang tao tu anh VALID..." -ForegroundColor Cyan
        & $python scripts/make_demo_video.py --split val --shuffle-seed 42
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Error "Khong tim thay video: $Source"
    }
}

$cmdArgs = @("scripts/predict_video.py", "--source", $Source, "--imgsz", $ImgSz, "--device", $Device)
if ($Weights -ne "") { $cmdArgs += @("--weights", $Weights) }
if ($Conf -gt 0) { $cmdArgs += @("--conf", $Conf) }
if ($MaxFrames -gt 0) { $cmdArgs += @("--max-frames", $MaxFrames) }
if ($NoTrack) { $cmdArgs += "--no-track" }
if ($View) { $cmdArgs += "--view" }
if ($Open) { $cmdArgs += "--open" }

Write-Host "Dang khoi chay nhan dien video voi lenh: python $($cmdArgs -join ' ')" -ForegroundColor Green
& $python @cmdArgs
exit $LASTEXITCODE
