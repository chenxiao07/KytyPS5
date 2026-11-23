$ErrorActionPreference = 'Stop'
# The Windows launcher's weave switch (LauncherWeaveTests.cs): its gate, its text and Play's command, no game files
# needed (Commands.PlayCommand with a test LauncherContext).
$root = Split-Path $PSScriptRoot -Parent
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$compiler = & $vswhere -latest -products '*' -requires Microsoft.Component.MSBuild -find 'MSBuild/Current/Bin/Roslyn/csc.exe'
if (!$compiler) { throw 'Visual Studio C# compiler not found.' }
$out = Join-Path $root '_Build/launcher-weave-tests'
New-Item -ItemType Directory -Force $out | Out-Null
$exe = Join-Path $out 'LauncherWeaveTests.exe'
$sources = @('Weave.cs', 'Mode4.cs', 'Settings.cs', 'Strings.cs', 'Commands.cs', 'Platform.cs', 'Game.cs', 'KeyboardPrompts.cs',
	'KeyboardLayout.cs', 'PromptTexture.cs', 'PageMarks.cs') | ForEach-Object { Join-Path $root "launcher-windows/$_" }
$arguments = @('/nologo', '/target:exe', '/r:System.Drawing.dll', '/r:System.Management.dll', '/r:System.Web.Extensions.dll', "/out:$exe") + $sources +
	@((Join-Path $PSScriptRoot 'LauncherWeaveTests.cs'))
& $compiler $arguments
if ($LASTEXITCODE -ne 0) { throw 'Test compilation failed.' }
& $exe $root
if ($LASTEXITCODE -ne 0) { throw 'Launcher weave tests failed.' }

# launcher.ps1's own gate (Test-WeaveOffered) and Play's choice between mode 4 and the weave: its functions taken from the
# script (not run: it opens its window), the memory and the CPUs given so nothing is read from this PC.
$tokens = $null
$errors = $null
$script = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'launcher.ps1'), [ref]$tokens, [ref]$errors)
$wanted = @('Get-Mode4Variable', 'ConvertFrom-CNumber', 'Test-Mode4Offered', 'Get-WeaveSwitch', 'Test-WeaveOffered')
foreach ($function in $script.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in $wanted }, $true)) {
	. ([scriptblock]::Create($function.Extent.Text))
}
foreach ($name in @('KYTY_SIMULATE_RAM_MB', 'KYTY_SIMULATE_PCORES', 'KYTY_SPECULATE_RAM_GB', 'KYTY_SPECULATE_THREADS', 'KYTY_SPECULATE_CPUS',
		'KYTY_SPECULATE', 'KYTY_WEAVE_WORKERS', 'KYTY_SCOUT', 'KYTY_VULKAN_RECORDING', 'KYTY_TABLE_XPR', 'KYTY_NATIVE_XPR')) {
	Remove-Item "env:$name" -ErrorAction SilentlyContinue
}
$failed = 0
function Test-Row([string]$what, $got, $want) {
	if ("$got" -ne "$want") { Write-Host "FAIL (launcher.ps1): $what = $got, not $want"; $script:failed++ }
}
# (Mode 4's own gate reads the memory and its CPUs from the -Set pairs here: KYTY_SIMULATE_RAM_MB, KYTY_SIMULATE_PCORES.)
function Test-Offered([string]$sets, [uint64]$performance, [uint64]$logical = 32) {
	return Test-WeaveOffered @(, @('-Set', "KYTY_SIMULATE_RAM_MB=16384,KYTY_SIMULATE_PCORES=16,$sets")) $performance $logical
}
Test-Row '10 CPUs' (Test-Offered 'A=1' 10) 'True'
Test-Row '9 CPUs' (Test-Offered 'A=1' 9) 'False'
Test-Row 'classes not known, 12 logical' (Test-Offered 'A=1' 0 12) 'True'
Test-Row 'classes not known, 8 logical' (Test-Offered 'A=1' 0 8) 'False'
Test-Row 'WEAVE_WORKERS=4 lifts it' (Test-Offered 'KYTY_WEAVE_WORKERS=4' 6) 'True'
Test-Row 'WEAVE_WORKERS=0 counts' (Test-Offered 'KYTY_WEAVE_WORKERS=0' 6) 'False'
Test-Row 'KYTY_SPECULATE=4, mode 4 stood down for memory' (Test-Offered 'KYTY_SPECULATE=4' 16) 'True'
Test-Row 'KYTY_SPECULATE=4 where mode 4 runs' (Test-Offered 'KYTY_SPECULATE=4,KYTY_SIMULATE_RAM_MB=32768' 16) 'False'
Test-Row 'KYTY_SCOUT=0' (Test-Offered 'KYTY_SCOUT=0' 16) 'False'
Test-Row 'an empty KYTY_TABLE_XPR is 0' (Test-Offered 'KYTY_TABLE_XPR=' 16) 'False'
Test-Row 'KYTY_SCOUT=2' (Test-Offered 'KYTY_SCOUT=2' 16) 'True'
$env:KYTY_NATIVE_XPR = '0'
Test-Row 'the environment read' (Test-Offered 'A=1' 16) 'False'
Test-Row '-Set over the environment' (Test-Offered 'KYTY_NATIVE_XPR=1' 16) 'True'
Remove-Item env:KYTY_NATIVE_XPR
if ($failed -ne 0) { throw "launcher.ps1's weave gate: $failed failed." }
Write-Host "launcher.ps1's weave gate passed."
