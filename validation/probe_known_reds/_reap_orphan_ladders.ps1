# Reap ONLY the orphaned ladder loops this session started and then stopped,
# and their direct pytest children.  Identified by explicit PID and by parent
# PID -- never by a command-line pattern (the launching shell shares it), and
# never anything else on this box.
param([int[]]$OrphanShellPids)

foreach ($p in $OrphanShellPids) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $p" -ErrorAction SilentlyContinue
    if (-not $proc) { Write-Output "shell $p : already gone"; continue }
    $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId = $p" -ErrorAction SilentlyContinue
    foreach ($k in $kids) {
        Write-Output ("  child {0} {1}" -f $k.ProcessId, $k.Name)
        # the timeout/python chain: reap grandchildren too
        $gk = Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $k.ProcessId) -ErrorAction SilentlyContinue
        foreach ($g in $gk) {
            Write-Output ("    grandchild {0} {1}" -f $g.ProcessId, $g.Name)
            try { Stop-Process -Id $g.ProcessId -Force -ErrorAction Stop } catch { Write-Output ("    grandchild stop failed: " + $_.Exception.Message) }
        }
        try { Stop-Process -Id $k.ProcessId -Force -ErrorAction Stop } catch { Write-Output ("  child stop failed: " + $_.Exception.Message) }
    }
    try { Stop-Process -Id $p -Force -ErrorAction Stop; Write-Output "shell $p : stopped" }
    catch { Write-Output ("shell " + $p + " stop failed: " + $_.Exception.Message) }
}
