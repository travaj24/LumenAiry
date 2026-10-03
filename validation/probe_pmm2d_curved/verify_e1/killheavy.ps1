for ($i = 0; $i -lt 360; $i++) {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'v5_pillar.py zstair (32 13|32 15|16 15) ' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "killed $($_.CommandLine)" }
  Start-Sleep -Seconds 10
}
