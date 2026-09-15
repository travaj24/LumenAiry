Get-CimInstance Win32_Process | ForEach-Object {
    if ($_.CommandLine -and ($_.CommandLine -match 'run_ladder' -or $_.CommandLine -match 'consumer_files')) {
        $cl = ($_.CommandLine -replace '\s+', ' ')
        if ($cl.Length -gt 140) { $cl = $cl.Substring(0, 140) }
        Write-Output ("{0}`t{1}`t{2}`t{3}" -f $_.ProcessId, $_.Name, $_.CreationDate, $cl)
    }
}
