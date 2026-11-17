# spike-walk.ps1: the 1-3 spike walk (tools/local/SPIKE-WALK.md). One boot: the baseline save in, the game up with the
# HUD, the shader prefetch finished, then -Walks times: archstone travel to 1-3, the player put at the walk's start facing
# down the corridor, the camera reset, an 18 s run onto the bridge with the frame windows measured, the flips dumped; the
# save restored at the end. Read the boots with tools/local/spike-walk-read.py.
#   powershell -File tools\local\windows\spike-walk.ps1 -Label base1 [-Exe <exe>] [-Mode 4] [-TravelBack] [-Set K=V,...]
# -Mode 4: KYTY_SPECULATE=4 KYTY_SCOUT=0; -Mode 0: neither. -Set: more launch switches (run-windows.ps1 -Set form).
# -TravelBack: before the walk, archstone travel to 1-1 and 70 s there, then back to 1-3 (the area's buffers and images
#   aged out by then: the cold form). -Tail N: N s of standing windows after the run. -Baseline <dir>: the save to start
#   from (bench-windows.py's KYTY_BENCH_BASELINE); it must have the 1-3 archstone (Boletarian Palace, Lord's Path).
# -Key <name>: KYTY_PREFETCH_CACHE_BUILD, a fixed shader-cache key, so the cache stays warm across builds.
param([Parameter(Mandatory = $true)][string]$Label, [string]$Exe = '', [ValidateSet(0, 4)][int]$Mode = 4,
      [string]$Set = '', [int]$Walks = 1, [switch]$TravelBack, [int]$Tail = 12, [string]$Baseline = '', [string]$Key = '',
      [string]$Out = '', [int]$StartupSeconds = 300)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
if (!$Out) { $Out = "$root\_Build\spike-walk" }
New-Item -ItemType Directory -Force $Out | Out-Null
$logs = "$root\_Build\run-logs"
$o = $Out -replace '\\', '/'
if (Get-Process kyty_emulator -ErrorAction SilentlyContinue) { 'STOP: an emulator is running'; exit 1 }
if ($Baseline) { $env:KYTY_BENCH_BASELINE = $Baseline }
python tools\local\bench-windows.py prepare
if ($LASTEXITCODE -ne 0) { 'STOP: bench-windows.py prepare failed'; exit 1 }
$sets = @('KYTY_FPS_HUD=2') + $(if ($Key) { @('KYTY_PREFETCH_CACHE=1', "KYTY_PREFETCH_CACHE_BUILD=$Key") } else { @() }) +
        $(if ($Mode -eq 4) { @('KYTY_SPECULATE=4', 'KYTY_SCOUT=0') } else { @() }) + $(if ($Set) { $Set -split ',(?=[A-Z_]+=)' } else { @() })
"== $Label (mode $Mode; $($sets -join ' ')) $(Get-Date -Format HH:mm:ss)"
$lines = @()
$bench = $null
try {
	$t0 = Get-Date
	$runArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$root\tools\local\windows\bench-run.ps1", '-Label', $Label,
	          '-KeepRunning', '-NoWalk', '-ShotOnly', '-Vblank', '120', '-StartupSeconds', "$StartupSeconds", '-Set', ($sets -join ','))
	if ($Exe) { $runArgs += @('-Exe', $Exe) }
	$bench = Start-Process powershell -ArgumentList $runArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$Out\$Label.bench.txt" -RedirectStandardError "$Out\$Label.bench.err.txt"
	# Up: the HUD (bench-run's "HUD after") and the shader prefetch finished ("Shader prefetch: all" in the log).
	$up = $false
	for ($w = 0; $w -lt 2 * ($StartupSeconds + 300) -and !$up; $w++) {
		Start-Sleep -Milliseconds 500
		$g = Get-ChildItem $logs -Filter '*.out.log' -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -ge $t0 } | Sort-Object CreationTime | Select-Object -Last 1
		$up = $g -and (Select-String -LiteralPath $g.FullName -Pattern '^Shader prefetch: all' -Quiet) -and
		      (Select-String -LiteralPath "$Out\$Label.bench.txt" -Pattern '^HUD after' -Quiet -ErrorAction SilentlyContinue)
		if (!$up -and $bench.HasExited -and !(Get-Process kyty_emulator -ErrorAction SilentlyContinue)) { break }
	}
	if (!$up) { "STOP: no HUD / shader prefetch ($((Get-Content "$Out\$Label.bench.txt" -ErrorAction SilentlyContinue | Select-Object -Last 2) -join ' | '))"; return }
	"[$Label] up at $(Get-Date -Format HH:mm:ss) ($([math]::Round(((Get-Date) - $t0).TotalSeconds)) s)"
	for ($i = 1; $i -le $Walks; $i++) {
		$p = "w$i"
		# The walk (SPIKE-WALK.md): the 1-3 archstone, the start with the facing set by `place` and read back by `where`, the
		# camera reset (R3, nothing to lock on), the enemies near the start killed (gamekill), the run.
		$walk = @('gamecmd +cp11_loadLevel 1 3', 'sleep 20')
		if ($TravelBack) { $walk += @('gamecmd +cp11_loadLevel 1 1', 'sleep 70', 'where', 'gamecmd +cp11_loadLevel 1 3', 'sleep 20') }
		$walk += @('place 17.25 47.999 455.5 -0.0635', 'sleep 0.3', 'where', 'pad 128 128 128 128 4 0.15', 'sleep 1.5', 'gamekill 300', 'sleep 1',
		           'pad 128 0 128 128 0 18', "measure 9 ${p}c00") + (1..28 | ForEach-Object { "measure 0.25 ${p}c{0:00}" -f $_ }) +
		         @("measure 2 ${p}c99") + $(if ($Tail -gt 0) { @(1..($Tail * 2) | ForEach-Object { "measure 0.5 ${p}t{0:00}" -f $_ }) } else { @() }) +
		         @('where', "flipsx 8192 $o/$Label-$p.flipsx.txt")
		$wo = @(python tools\local\bench-windows.py live ($walk -join ' ; '))
		$lines += $wo
		"[$Label] walk $i done $(Get-Date -Format HH:mm:ss)"
		if (!(Get-Process kyty_emulator -ErrorAction SilentlyContinue) -or ($wo -match 'not running')) { "[$Label] CRASH during walk $i"; break }
		if (@($wo | Select-String 'LIVE_MEASURE .*frames=0 ').Count -ge 3) { "[$Label] FREEZE in walk $i"; break }
	}
} finally {
	$lines | Set-Content "$Out\$Label.live.txt" -Encoding ascii
	$g = Get-ChildItem $logs -Filter '*.out.log' -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -ge $t0 } | Sort-Object CreationTime | Select-Object -Last 1
	if ($g) { Copy-Item $g.FullName "$Out\$Label.out.log" }
	Get-Process kyty_emulator -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
	if ($bench) { Get-Process powershell -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq $bench.Id } | Stop-Process -Force -ErrorAction SilentlyContinue }
	for ($w = 0; $w -lt 30 -and (Get-Process kyty_emulator -ErrorAction SilentlyContinue); $w++) { Start-Sleep 1 }
	python tools\local\bench-windows.py restore 2>&1 | Select-Object -Last 1
	Remove-Item env:KYTY_BENCH_BASELINE -ErrorAction SilentlyContinue
	"[$Label] measures: $(@($lines | Select-String 'LIVE_MEASURE').Count), files in $Out $(Get-Date -Format HH:mm:ss)"
}
