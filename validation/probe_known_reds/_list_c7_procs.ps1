Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | ForEach-Object {
    if ($_.CommandLine -match 'test_niche_c7_ray_density_halo_check') {
        Write-Output ("{0}`t{1}`t{2}" -f $_.ProcessId, $_.CreationDate, $_.WorkingSetSize)
    }
}
