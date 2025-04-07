@echo off
REM Projektstarter für AP-LLM

REM Aktiviere virtuelle Umgebung
call .venv\Scripts\activate

REM Installiere Abhängigkeiten
pip install -r requirements.txt

REM Starte Visual Studio Code im Projektverzeichnis
code .

echo -----------------------------------------
echo Projektumgebung wurde gestartet!
echo Öffne nun eine Python-Datei oder Terminal.
pause
