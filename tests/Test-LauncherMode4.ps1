$ErrorActionPreference = 'Stop'
# The Windows launcher's parallel translation switch (LauncherMode4Tests.cs): its gate, its text and Play's command, no
# game files needed (Commands.PlayCommand with a test LauncherContext).
$root = Split-Path $PSScriptRoot -Parent
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$compiler = & $vswhere -latest -products '*' -requires Microsoft.Component.MSBuild -find 'MSBuild/Current/Bin/Roslyn/csc.exe'
if (!$compiler) { throw 'Visual Studio C# compiler not found.' }
$out = Join-Path $root '_Build/launcher-mode4-tests'
New-Item -ItemType Directory -Force $out | Out-Null
$exe = Join-Path $out 'LauncherMode4Tests.exe'
$sources = @('Mode4.cs', 'Settings.cs', 'Strings.cs', 'Commands.cs', 'Platform.cs', 'Game.cs', 'KeyboardPrompts.cs', 'KeyboardLayout.cs',
	'PromptTexture.cs', 'PageMarks.cs') | ForEach-Object { Join-Path $root "launcher-windows/$_" }
$arguments = @('/nologo', '/target:exe', '/r:System.Drawing.dll', '/r:System.Management.dll', '/r:System.Web.Extensions.dll', "/out:$exe") + $sources +
	@((Join-Path $PSScriptRoot 'LauncherMode4Tests.cs'))
& $compiler $arguments
if ($LASTEXITCODE -ne 0) { throw 'Test compilation failed.' }
& $exe $root
if ($LASTEXITCODE -ne 0) { throw 'Launcher mode 4 tests failed.' }

# launcher.ps1's own gate (Test-Mode4Offered): its functions taken from the script (not run: it opens its window), the
# memory and the performance cores simulated so nothing is read from this PC.
$tokens = $null
$errors = $null
$script = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'launcher.ps1'), [ref]$tokens, [ref]$errors)
$wanted = @('Get-Mode4Variable', 'ConvertFrom-CNumber', 'Test-Mode4Offered', 'Get-PerformanceCpus')
foreach ($function in $script.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in $wanted }, $true)) {
	. ([scriptblock]::Create($function.Extent.Text))
}
foreach ($name in @('KYTY_SIMULATE_RAM_MB', 'KYTY_SIMULATE_PCORES', 'KYTY_SPECULATE_RAM_GB', 'KYTY_SPECULATE_THREADS', 'KYTY_SPECULATE_CPUS')) {
	Remove-Item "env:$name" -ErrorAction SilentlyContinue
}
$failed = 0
function Test-Row([string]$what, $got, $want) {
	if ("$got" -ne "$want") { Write-Host "FAIL (launcher.ps1): $what = $got, not $want"; $script:failed++ }
}
function Test-Offered([string]$sets) { return Test-Mode4Offered @(, @('-Set', $sets)) }
Test-Row '40000MB' ((ConvertFrom-CNumber '40000MB' ([uint64]::MaxValue)) -join ' ') '40000 False'
Test-Row "' 12'" ((ConvertFrom-CNumber ' 12' ([uint64][uint32]::MaxValue)) -join ' ') '12 True'
Test-Row 'x' ((ConvertFrom-CNumber 'x' ([uint64]::MaxValue)) -join ' ') '0 False'
Test-Row '-1 (strtoul)' ((ConvertFrom-CNumber '-1' ([uint64][uint32]::MaxValue))[0]) '4294967295'
Test-Row '32 GiB, 16 CPUs' (Test-Offered 'KYTY_SIMULATE_RAM_MB=32768,KYTY_SIMULATE_PCORES=16') 'True'
Test-Row '16 GiB' (Test-Offered 'KYTY_SIMULATE_RAM_MB=16384,KYTY_SIMULATE_PCORES=16') 'False'
Test-Row 'RAM_GB=0' (Test-Offered 'KYTY_SIMULATE_RAM_MB=16384,KYTY_SIMULATE_PCORES=16,KYTY_SPECULATE_RAM_GB=0') 'True'
Test-Row 'RAM_GB=12x' (Test-Offered 'KYTY_SIMULATE_RAM_MB=16384,KYTY_SIMULATE_PCORES=16,KYTY_SPECULATE_RAM_GB=12x') 'False'
Test-Row 'an empty RAM_MB is 0' (Test-Offered 'KYTY_SIMULATE_RAM_MB=,KYTY_SIMULATE_PCORES=16') 'False'
Test-Row '8 CPUs' (Test-Offered 'KYTY_SIMULATE_RAM_MB=32768,KYTY_SIMULATE_PCORES=8') 'False'
Test-Row '8 CPUs, 3 workers' (Test-Offered 'KYTY_SIMULATE_RAM_MB=32768,KYTY_SIMULATE_PCORES=8,KYTY_SPECULATE_THREADS=3') 'True'
Test-Row '4 CPUs, CPUS named' (Test-Offered 'KYTY_SIMULATE_RAM_MB=32768,KYTY_SIMULATE_PCORES=4,KYTY_SPECULATE_CPUS=8-15') 'True'
$env:KYTY_SIMULATE_RAM_MB = '16384'
Test-Row 'the environment read' (Test-Offered 'KYTY_SIMULATE_PCORES=16') 'False'
Test-Row '-Set over the environment' (Test-Offered 'KYTY_SIMULATE_PCORES=16,KYTY_SIMULATE_RAM_MB=32768') 'True'
Remove-Item env:KYTY_SIMULATE_RAM_MB
if ($failed -ne 0) { throw "launcher.ps1's mode 4 gate: $failed failed." }
Write-Host "launcher.ps1's mode 4 gate passed."
