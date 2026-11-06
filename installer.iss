#define MyAppName "EcoAcoustic Sentinel"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "EcoAcoustic Sentinel Team"
#define MyAppURL "https://github.com/"
#define MyAppExeName "EcoAcousticSentinel.exe"
#define BuildRoot "dist\\EcoAcousticSentinel"
#define BuildExe "dist\\EcoAcousticSentinel\\EcoAcousticSentinel.exe"
#define SetupIcon "assets\\app.ico"

#ifnexist BuildExe
  #error "No se encontro dist\\EcoAcousticSentinel\\EcoAcousticSentinel.exe. Ejecuta primero: python build_exe.py"
#endif

[Setup]
AppId={{73DA4E2E-A3C2-4CB1-9E3E-CB3B5A8BEFA7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\EcoAcousticSentinel
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
OutputDir=dist
OutputBaseFilename=EcoAcousticSentinel_Installer_x64
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupLogging=yes
#ifexist SetupIcon
SetupIconFile={#SetupIcon}
#endif

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el escritorio"; GroupDescription: "Tareas adicionales:"

[Files]
Source: "{#BuildRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\profiles"
Name: "{app}\output"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Ejecutar {#MyAppName}"; Flags: nowait postinstall skipifsilent
