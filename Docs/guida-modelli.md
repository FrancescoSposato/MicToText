# Guida ai modelli di MicToText

**Aggiornata al:** 21 settembre 2026
**Hardware:** ASUS TUF — RTX 5060 Laptop (8 GB VRAM), Ryzen 7 260, 16 GB RAM

Questa guida descrive i modelli **effettivamente installati** su questa macchina, cosa fanno
nell'app, dove eccellono e dove falliscono.

Quasi tutti i dati vengono da **misure fatte sul campo** durante lo sviluppo, sulle tue
registrazioni e su video reali. Dove un'informazione viene invece dalla documentazione o da
benchmark esterni, è indicato. Diverse misure sono state prese **una sola volta**, su contenuti
brevi: sono indicazioni solide sull'ordine di grandezza, non statistiche.

---

## In breve

| Fase | Modello consigliato | Alternativa |
|---|---|---|
| Trascrizione | **Whisper large-v3-turbo** (GPU) | Whisper small (CPU, ripiego automatico) |
| Appunti | **qwen3.5:9b** con ragionamento | — |
| Schemi logici | **qwen3.5:9b** senza ragionamento | — |
| Schede concetto | **qwen3.5:9b** | — |
| Rigenerazione mirata | **qwen3.5:9b** | — |
| Resa grafica | mermaid-cli (non è un modello) | — |

**In sintesi:** `qwen3.5:9b` è superiore a `dolphin3:8b` su tutto ciò che conta per questa app.
Il motivo principale è che `dolphin3:8b` **inventa contenuti**, e per uno strumento di studio è il
difetto peggiore possibile.

---

## Quale modello interviene in quale fase

| Fase | Tipo di modello | Configurazione |
|---|---|---|
| Registrazione | nessuno | cattura audio con `sounddevice` |
| Filtro pause e rumore | nessuno | regole sui segnali di Whisper, nessun modello |
| **Trascrizione** | Whisper | *Modello Whisper* nel Controllo avanzato |
| **Appunti** | LLM | *Modello Ollama (appunti)* |
| **Schema logico** | LLM | *Modello Ollama (schema)*, altrimenti usa quello degli appunti |
| **Schemi per argomento** | LLM | come sopra |
| **Scelta dei concetti** | LLM | come sopra |
| **Schede concetto** | LLM | come sopra |
| **Rigenerazione** | LLM | come sopra |
| Resa degli schemi | nessuno | mermaid-cli + Chromium headless |

Il ragionamento interno (*thinking*) si regola a parte, con **Ragionamento del modello** nelle
impostazioni. Pesa più della scelta del modello stesso: vedi la scheda di `qwen3.5:9b`.

---

## Modelli di trascrizione (Whisper)

Tutti in versione `faster-whisper` (CTranslate2), eseguiti in un processo separato che libera la
VRAM appena finisce.

### Whisper large-v3-turbo — il predefinito

| | |
|---|---|
| **Uso** | Trascrizione su GPU |
| **Dimensione** | 1,6 GB su disco, circa 2 GB di VRAM (float16) |
| **Stato** | ✅ Scaricato |

**Punti di forza**

- **Velocissimo** sulla tua GPU. Misurato:

  | Audio | Tempo di trascrizione |
  |---|---|
  | 19 s | 4,9 s |
  | 29 s | 4,8 s |
  | 281 s | 13,4 s |

  Il costo fisso di caricamento del modello pesa più della durata dell'audio: su file brevi
  il tempo quasi non cambia.
- **Italiano quasi perfetto** su tutto il materiale provato: lezioni, video, registrazioni dal
  microfono.
- Qualità dichiarata molto vicina a `large-v3`, con il decoder ridotto da 32 a 4 livelli
  (fonte: documentazione OpenAI e benchmark esterni).

**Punti deboli**

- **Nomi propri**: `Lubicz` è diventato `Lubic`, `Roma Tre` è diventato `Roma III`.
- **Parole poco chiare**: in inglese `trunks` (proboscide) è diventato `fronts`, e l'errore si è
  poi propagato negli appunti.
- Occupa circa 2 GB di VRAM, che **si sommano** al modello Ollama se lanci due sessioni ravvicinate
  (vedi *Convivenza in VRAM*).

### Whisper small — il ripiego

| | |
|---|---|
| **Uso** | Trascrizione su CPU, attivata in automatico se la GPU fallisce |
| **Dimensione** | 464 MB, in int8 |
| **Stato** | ✅ Scaricato |

**Punti di forza**

- Funziona **senza GPU**: è la rete di sicurezza della pipeline.
- Leggero, utile per prove rapide dei meccanismi (sezioni, filtro) dove la qualità del testo
  non conta.

**Punti deboli**

- Sensibilmente meno accurato di `large-v3-turbo`, soprattutto su termini tecnici.
- Su CPU è molto più lento, su registrazioni lunghe la differenza è netta.

### Gli altri Whisper del menu

`tiny`, `base`, `medium` e `large-v3` compaiono nel menu ma **non sono scaricati**: vengono
scaricati da Hugging Face al primo utilizzo (circa 75 MB, 145 MB, 1,5 GB e 3 GB).

| Modello | Quando ha senso |
|---|---|
| `tiny`, `base` | Solo prove dei meccanismi. Troppo imprecisi per appunti di studio |
| `medium` | Poco: `large-v3-turbo` è quasi sempre migliore e non più lento |
| `large-v3` | Massima accuratezza possibile, ma molto più lento di turbo e con circa il doppio della VRAM. Da provare solo se turbo sbaglia sistematicamente su un certo materiale |

---

## Modelli di linguaggio (Ollama)

### qwen3.5:9b — il consigliato

| | |
|---|---|
| **Uso** | Appunti, schemi, schede, rigenerazione: tutte le fasi testuali |
| **Dimensione** | 9,7 miliardi di parametri, quantizzazione Q4_K_M, 6,6 GB su disco |
| **VRAM** | Circa 5,5 GB caricato, interamente su GPU (`100% GPU` in `ollama ps`) |
| **Contesto massimo** | 262.144 token (in pratica limitato dalla VRAM) |
| **Capacità** | testo, **ragionamento**, **visione**, strumenti |
| **Stato** | ✅ Installato |

**Punti di forza**

- **Non inventa, col ragionamento attivo.** È il risultato più importante di tutta la sessione di
  sviluppo. Su un video di 30 secondi che si limitava a porre una domanda, ha scritto appunti in
  cui ogni affermazione era verificabile nella trascrizione, e ha riconosciuto correttamente che il
  video era solo un'introduzione. `dolphin3:8b`, sullo stesso video, si era inventato una lezione
  di fisica intera.
- **Segue i vincoli del prompt.** Usa come radice dello schema la *domanda* a cui il contenuto
  risponde, etichetta tutti gli archi con verbi che si leggono come frasi, non trasforma in nodi
  le sezioni strutturali degli appunti.
- **Omette ciò che non sa.** Nelle schede ha saltato il blocco *Attenzione* quando gli appunti non
  offrivano un errore tipico da segnalare, invece di inventarne uno.
- **Segnala le proprie incertezze**: ha scritto *"la 'cazia migliore' (probabilmente riferita a…)"*
  invece di trasformare una parola trascritta male in un concetto inventato.
- **Capacità di visione non ancora sfruttata**: potrebbe guardare gli schemi già renderizzati e
  giudicarne la leggibilità. Oggi l'app non la usa.

**Punti deboli**

- **Il ragionamento è lentissimo.** Su un prompt banale, misurato:

  | | Tempo | Testo di ragionamento prodotto |
  |---|---|---|
  | Con ragionamento | 115,3 s | 17.454 caratteri |
  | Senza | 0,4 s | 0 |

- **Senza ragionamento torna a inventare.** Sugli stessi 30 secondi ha aggiunto relatività
  generale, potenziale gravitazionale, orologi atomici: nulla di questo era nel video. È per
  questo che l'app applica il ragionamento **solo agli appunti**.
- **Ignora il campo *argomenti corretti*** nella rigenerazione: due volte su due ha eliminato una
  distinzione che gli era stato chiesto di conservare, anche dichiarandola come vincolo non
  negoziabile. L'app ora controlla meccanicamente i termini persi e chiede una correzione.
- Il più pesante dei due: vedi *Convivenza in VRAM*.

**Tempi misurati sulla stessa sorgente, per livello di ragionamento**

| Livello | Appunti | Schema | 2 schede | Totale | Fedeltà |
|---|---|---|---|---|---|
| Nessuno | 26 s | 21 s | 24 s | **1 min 22 s** | ❌ inventa |
| **Solo appunti** (predefinito) | 82 s | 17 s | 25 s | **2 min 12 s** | ✅ |
| Completo | 119 s | 46 s | 587 s | **12 min 42 s** | ✅ |

Il livello *Completo* costa sei volte tanto senza benefici misurabili: le schede e gli schemi
rielaborano appunti già scritti, e lì il ragionamento non aggiunge fedeltà.

### dolphin3:8b — sconsigliato per i contenuti

| | |
|---|---|
| **Uso** | Alternativa più leggera; era il predefinito all'inizio dello sviluppo |
| **Dimensione** | 8 miliardi di parametri (famiglia Llama), Q4_K_M, 4,9 GB |
| **Contesto massimo** | 131.072 token |
| **Capacità** | solo testo: niente ragionamento, niente visione |
| **Stato** | ✅ Installato |

**Punti di forza**

- **Più leggero**: 4,9 GB lasciano spazio a Whisper nella VRAM, quindi non c'è conflitto fra
  sessioni ravvicinate.
- **Rapido**: gli schemi escono in circa 20-24 secondi.
- **Buona prosa**: i blocchi di testo delle prime schede, generati da lui, erano apprezzati.
- Adatto a **provare i meccanismi dell'app** (interfaccia, sezioni, filtro, rinomina delle
  cartelle), quando il contenuto degli appunti non interessa.

**Punti deboli** — e sono gravi

- **Inventa contenuti.** Da un video che poneva solo una domanda ha prodotto l'esperimento di
  Pound-Rebka, la curvatura dello spaziotempo e le velocità prossime a quella della luce. Tutte
  cose vere in fisica, e proprio per questo pericolose: studiandole non le avresti riconosciute
  come inventate.
- **Ignora i vincoli del prompt.** Ha trasformato in nodi le sezioni *Key terms* e *Open
  questions* per tre volte di fila nonostante un divieto esplicito, e ha scelto uno schema
  orizzontale quando era richiesto verticale, producendo una striscia illeggibile in rapporto 12:1.
- **Sintassi Mermaid fragile.** Dimentica di chiudere i gruppi (fino a 3 per schema), crea gruppi
  vuoti che fanno fallire il motore di impaginazione, dichiara i colori ma non li assegna. L'app
  corregge in automatico i primi due difetti, ma sono segnali di scarsa affidabilità.
- **Non etichetta le relazioni**: schemi con archi muti, che non spiegano nulla.
- **Inventa parole**: ha intitolato uno schema *"Il Tempo e le Sue Paradoxosità"*, termine che
  non esiste in italiano.
- Non supporta il ragionamento, quindi non esiste un livello che lo renda fedele.

---

## I migliori per attività

### Studiare una lezione registrata

**Whisper large-v3-turbo + qwen3.5:9b, ragionamento *Solo appunti*.**
È la configurazione predefinita, e la più equilibrata fra fedeltà e tempo.

### Contenuti in cui un errore costa caro (esami, argomenti tecnici)

**Come sopra, e rileggi gli appunti confrontandoli con `trascrizione.txt`.**
Nessun modello locale è infallibile. Se Whisper sbaglia i termini tecnici di una materia, prova
`large-v3`.

### Prove rapide dell'interfaccia o dei parametri

**Whisper small + qwen3.5:9b, ragionamento *Nessuno*, 0 schede.**
Qualche decina di secondi per ciclo. Il testo prodotto non è affidabile, ma per verificare come
cambia uno schema spostando un cursore basta e avanza.

### Lezioni lunghe (oltre un'ora)

**qwen3.5:9b, con Blocchi trascrizione e Finestra di contesto alzati insieme.**
Oltre 12.000 caratteri di trascrizione, gli appunti vengono generati a parti e poi fusi, e la
fusione può impoverire i concetti a cavallo fra due parti. Portare i blocchi a 20.000 e il contesto
a 32k riduce le fusioni; se `ollama ps` mostra una parte su CPU, torna a 16k.

### Sessioni ravvicinate, una dopo l'altra

**Due strade:** usare `dolphin3:8b`, oppure fermare il modello fra una sessione e l'altra con
`ollama stop qwen3.5:9b`. Vedi il paragrafo seguente.

### Molte schede concetto (il punto di forza dell'app)

**qwen3.5:9b senza ragionamento sulle schede**, cioè il livello predefinito.
Le schede partono da appunti già scritti e verificati: il ragionamento lì costa minuti senza
migliorare la fedeltà. Sono la parte dell'output con il miglior rapporto fra valore e tempo.

---

## Convivenza in VRAM

Gli 8 GB sono il vincolo principale. Dentro una singola sessione non ci sono conflitti: Whisper
libera la memoria prima che parta il modello di linguaggio.

Il problema si presenta **fra una sessione e la successiva**, perché Ollama tiene il modello
caricato per 10 minuti (`keep_alive`).

| Modello caricato | + Whisper turbo | Esito |
|---|---|---|
| `dolphin3:8b` | circa 7 GB | ✅ ci sta |
| `qwen3.5:9b` | circa 8,2 GB | ⚠️ al limite: Whisper rischia di finire su CPU |

Durante una generazione di `qwen3.5:9b` la GPU occupava circa 6,7 GB in tutto. Se lanci una seconda
sessione entro 10 minuti e noti la trascrizione rallentata, è questo il motivo. Rimedi:
`ollama stop qwen3.5:9b` prima della nuova sessione, oppure ridurre `keep_alive` in `config.py`.

---

## Non installati, da valutare

Dalla ricerca del 17 settembre (vedi `ricerca-modelli-2026.md`). **Nessuno di questi è stato
provato** nell'app.

| Modello | Perché potrebbe interessare | Come ottenerlo |
|---|---|---|
| **NVIDIA Parakeet TDT 0.6B v3** | Trascrizione dichiarata più accurata di Whisper large-v3 in italiano, e più veloce | Richiede integrazione: libreria `onnx-asr` |
| `qwen2.5-coder:7b` | Sintassi Mermaid più precisa | `ollama pull qwen2.5-coder:7b` |
| `gemma4:e4b` | Alternativa multilingue più leggera | `ollama pull gemma4:e4b` |
| `ministral-3:8b` | Mistral è forte sulle lingue europee | `ollama pull ministral-3:8b` |

Il candidato con il ritorno potenziale più alto è **Parakeet**: agirebbe sulla trascrizione, e un
errore di trascrizione si propaga ad appunti, schemi e schede.

---

## Come sono state ottenute queste informazioni

- **Misurate** in questa sede: tempi, uso della VRAM, fedeltà degli appunti, rispetto dei vincoli,
  errori di sintassi. Le fonti sono le sessioni in `output/` e i test eseguiti durante lo sviluppo.
- **Da documentazione o benchmark esterni**: le qualità dichiarate dei modelli non installati e il
  confronto teorico fra `large-v3` e `large-v3-turbo`.
- **Limite principale**: molti confronti sono stati fatti una volta sola e su contenuti brevi
  (30 secondi – 5 minuti). Su una lezione vera di un'ora i rapporti fra i modelli potrebbero
  cambiare, e vale la pena riverificarli.
