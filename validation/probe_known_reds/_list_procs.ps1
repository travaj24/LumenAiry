$c = 0
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | ForEach-Object {
    if ($_.CommandLine -match 'test_analytic_ray_transfer|test_audit2609_a15b_system_sub') {
        $c++
        $cl = ($_.CommandLine -replace '\s+', ' ')
        if ($cl.Length -gt 130) { $cl = $cl.Substring(0, 130) }
        Write-Output ("{0}`t{1}" -f $_.ProcessId, $cl)
    }
}
Write-Output ("MATCHED: " + $c)
