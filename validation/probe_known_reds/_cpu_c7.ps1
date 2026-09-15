Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | ForEach-Object {
    if ($_.CommandLine -match 'test_niche_c7_ray_density_halo_check') {
        $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        Write-Output ("{0}`tCPU={1}s`tWS={2}MB`tstart={3}" -f $_.ProcessId,
            [math]::Round($p.CPU, 1),
            [math]::Round($p.WorkingSet64 / 1MB),
            $_.CreationDate)
    }
}
