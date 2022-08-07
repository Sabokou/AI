# Allgemeines
- Matrikelnummer: 5920414
- Vorlesung: Künstliche Intelligenz
- Einreichungsform: Implementierung mit Quellcode
- Themenkomplex: 3

# Nutzungshinweise
Diese Implementierung eines reinforcement learning Ansatzes wurde mittels _Ray_ und OpenAI's _Gym_ umgesetzt.  
Die run.py beinhaltet das Trainingsskript für ein Proximal Policy Optimization Algorithmus. Es kann jedoch aufgrund der Flexibilität
von Ray Tune auch für andere kompatible Algorithmen genutzt werden. DQNs sind leider nicht für die Environment unterstützt,
da mehrere diskrete Aktionen im Action Space durchgeführt werden müssen. Eine Liste aller zur Verfügung stehenden Algorithmen
ist zu finden unter https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#available-algorithms-overview.

Um die Environment zu nutzen, muss diese lokal installiert werden, dafür sei folgender Befehl auszuführen:
```bash
pip install -e gridworld-strassenbahnkreuzung-gym
```
Alle anderen Pakete lassen sich durch 
```bash
pip install "ray[tune]" "ray[rllib]" tensorflow
```
installieren. 

Das Problem mit Ray als Bibliothek für Reinforcement Learning ist, dass es für lokale Tests ohne verteilte Hardware nur beschränkt
nutzbar ist. Ich habe es nicht hingekriegt in Google Colab die angebotene CUDA-fähige Grafikkarte zum Training zu nutzen.


Zur Visualisierung der Ergebnisse werden die Ergebnisse im logs/ Ordner gespeichert. In der Abgabe sind 2 exemplarisch trainierte
PPO-Agenten. Unter Reward ist zu sehen, dass die Algorithmen bereits innerhalb weniger Steps eine brauchbare Policy entwickeln und diese im weiteren
Verlauf sowohl schlechter als auch wieder besser wird. 
Tensorboard lässt sich nach Installation der Pip-Pakete über
```bash
tensorboard --logdir logs
``` 
starten.

# Motivation
In der realen Welt ist die Disposition von Zügen eines der kritischsten Aufgaben von Fahrdienstleitern. Disposition bedeutet
hier vereinfach dargestellt, welcher Zug vor welchem Zug auf einer Strecke fahren darf. In der erstellten Environment
wurde eine vereinfachte Version eines Streckennetzes gebaut mit zwei Kreuzungen. Die Aufgabe des Agenten ist es basierend auf der
Verspätung von Zügen zu entscheiden, welcher Zug Vorfahrt haben soll. Dafür kann er in jedem Step nur ein Signal pro Kreuzung auf _Grün_
schalten. Als Vereinfachungen wurde u.a. die Welt als Gridworld dargestellt und die Weichen werden durch die Züge bedient.
Vor allem letzteres ist sehr realitätsfremd, da dort Weichenstellung zum dispositiven Arbeitsfeld des Fahrdienstleiters gehört.
Die erste Vereinfachung ist jedoch nicht vollkommen realitätsfern, da in der echten Zugfahrt Zugstrecken in Blöcke (von Signal
zu Signal) unterteilt sind und sich jeweils nur ein Zug auf einem solchen Block befinden kann.

Als Reward wird nach jeder Kreuzung basierend auf der Verspätung des Zuges ein Reward zurückgegeben. Höhere Verspätungen
werden exponentiell bestraft. 

Eines der größten Lernfelder, die ich aus diesem Projekt mitgenommen habe, war die Arbeit und die Erstellung von eigenen OpenAI Gyms.
Vor allem die Registrierung und der Paketaufbau waren dabei neue Lernfelder. Auch war es spannend mit einer "industry-grade"
Reinforcement Learning Bibliothek zu arbeiten, welches seinerseits Troubleshooting sowohl erleichterte (bessere Fehlermeldungen)
als auch erschwerte, da die aktive Community kleiner ist und weniger Probleme auf StackOverflow o.ä. dokumentiert sind.

