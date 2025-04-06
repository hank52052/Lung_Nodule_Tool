Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' 獲取當前 .vbs 所在的資料夾
currentPath = FSO.GetParentFolderName(WScript.ScriptFullName)

' 設定 .bat 檔案的相對路徑
batPath = currentPath & "\dependence\Nodule_Tool.bat"

' 切換到目標資料夾，並執行 .bat，確保不彈出黑窗
WshShell.Run "cmd.exe /c cd /d " & """" & currentPath & "\dependence""" & " && " & """" & batPath & """", 0, False
