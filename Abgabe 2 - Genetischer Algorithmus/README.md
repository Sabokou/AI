# Allgemeines
- Matrikelnummer: 5920414
- Vorlesung: Künstliche Intelligenz
- Einreichungsform: Implementierung mit Quellcode
# Nutzungshinweise
Für das Skript werden diverse Pakete benötigt. Diese können über 
```bash
pip install -r requierements.txt
```
installiert werden. Es wird garantiert, dass die angegebenen Paketversionen für Python-Version >= 3.9 funktionieren.

Das Skript ist angedacht unter Nutzung von Parametern in der Konsole aufgerufen zu werden. Verpflichtend sind somit für die Nutzung,
dass mindestens die Parameter Populationsgröße und Featureanzahl bei Start mit übergeben werden analog zu folgendem Befehl
```bash
python '.\Genetischer Algorithmus.py' [Population] [Featurezahl]
```
Weitergehend verfügt das Skript noch über diverse optionale Parameter, um die einzelnen Phasen der Entwicklung anzupassen.
Dafür kann der oben aufgezeigte Befehl mit dem Argument --help / -h aufgerufen werden.
```bash
python '.\Genetischer Algorithmus.py' -h
``` 
Eine genaue Erklärung der angebotenen Methoden
ist der Dokumentation im Quellcode zu entnehmen.
# Motivation
Das Skript simuliert ein genetisches Verfahren, um Individuen zu züchten, deren Fitness gegeben durch ihre Chromosome zu maximieren
Dabei können die Gene der Individuen entweder _0_ oder _1_ annehmen. Als Fitnessfunktion wird die Summe aller Gene genutzt.
Alternative Grundbeispiele, um ein solches Verfahren zu demonstrieren wären u.a. das Problem des Handelsreisenden oder die Approximation von
höher dimensionierten oder anderen Exponentialfunktionen durch im genetisch Verfahren bestimmten Koeffizienten einer polynomialen Funktion.
Das hier umgesetzte Beispiel wurde inspiriert durch https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3 und ergänzt um Methoden und Verfahren aus
https://ls11-www.cs.tu-dortmund.de/lehre/SoSe03/PG431/Ausarbeitungen/GA_Selzam.pdf .