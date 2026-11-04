# 编译 cuda_mlp.cu：nvcc 需要 MSVC 的 cl.exe + Windows SDK 头文件。
# 本机 Windows SDK 装在 E:\Windows Kits\10，而 VS 的 DevShell 没自动带上（注册表查询被拦），
# 所以这里手动补 INCLUDE / LIB。
$ErrorActionPreference = 'Continue'
Import-Module "E:\vs2022\社区版\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath "E:\vs2022\社区版" -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$sdk = 'E:\Windows Kits\10'
$v = '10.0.26100.0'
$msvc = 'E:\vs2022\社区版\VC\Tools\MSVC\14.44.35207'
$env:INCLUDE = "$msvc\include;$sdk\Include\$v\ucrt;$sdk\Include\$v\um;$sdk\Include\$v\shared;$sdk\Include\$v\winrt"
$env:LIB = "$msvc\lib\x64;$sdk\Lib\$v\ucrt\x64;$sdk\Lib\$v\um\x64"
& "E:\cpp_project\.cuda-env\Library\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_89 --use_fast_math `
	--allow-unsupported-compiler `
	-o "E:\cpp_project\CppMLP\cuda\cuda_mlp.exe" "E:\cpp_project\CppMLP\cuda\cuda_mlp.cu" 2>&1 |
	Select-Object -Last 10
"=== nvcc exit: $LASTEXITCODE ==="
