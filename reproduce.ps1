# Use the active Python (e.g. a venv) if available; fall back to "python" on PATH.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "python" }
$src = Join-Path $PSScriptRoot "src"
$steps = @(
  "step1_build_windows.py",
  "step2_pc_gmm.py",
  "step3_features.py",
  "step4_loso_classical.py",
  "step5_loso_deep.py",
  "step6_ablation.py",
  "step7_confusion_compare.py",
  "step8_pca.py",
  "step9_posthoc.py",
  "step10_int8.py",
  "step11_signal_equivalence.py",
  "step12_figure1.py"
)
$failed = 0
foreach ($s in $steps) {
  Write-Output "===== $s ====="
  & $py (Join-Path $src $s)
  $code = $LASTEXITCODE
  Write-Output "exit=$code"
  if ($code -ne 0) { $failed = $failed + 1 }
}
Write-Output "ALL_DONE failed_steps=$failed"
