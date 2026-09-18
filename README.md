# MicToText

Pipeline locale da riga di comando: microfono → trascrizione (faster-whisper) →
appunti strutturati in Markdown (Ollama) → schema Mermaid (Ollama) → immagine
(mermaid-cli, con fallback HTML).

Nessuna chiamata a servizi cloud: tutto gira sulla macchina locale, salvo il
download iniziale dei pesi del modello Whisper e dei modelli Ollama.

## Rendering Mermaid: nativo su Windows, WSL2 come fallback

`mmdc` (mermaid-cli) è già installato nativamente su questa macchina
(`npm install -g @mermaid-js/mermaid-cli`, con Chromium scaricato da
Puppeteer in `%USERPROFILE%\.cache\puppeteer`). Con `--renderer auto`
(il default) l'app usa **prima questa installazione nativa**, senza passare
da WSL2: niente avvio della VM, niente conversione di percorsi.

Se in futuro il rendering nativo fallisce con l'errore
`Failed to launch the browser process: Code: 3221225595`
(`0xC000007B`, `STATUS_INVALID_IMAGE_FORMAT`), la causa è quasi sempre una
build di Chromium/`chrome-headless-shell` scaricata corrotta o incompleta
nella cache di Puppeteer (capitato una volta: mancava proprio `chrome.exe`,
probabilmente per un intervento dell'antivirus durante l'estrazione).
Si risolve reinstallando la build:

```powershell
npx --yes @puppeteer/browsers install chrome-headless-shell@150.0.7871.24 --path "$env:USERPROFILE\.cache\puppeteer"
```

(la versione esatta richiesta la trovi nel messaggio d'errore di `mmdc`, es.
`Could not find chrome-headless-shell (ver. X.Y.Z)`). WSL2 resta disponibile
come fallback automatico (`--renderer wsl`) se quello nativo non è
utilizzabile.

Per l'analisi architetturale completa, le istruzioni di installazione passo
passo e i limiti noti, vedi la documentazione di progetto consegnata insieme
al codice (analisi dei componenti, scelte motivate, troubleshooting).

## Avvio rapido

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-gpu.txt   # oppure requirements.txt per solo CPU

ollama pull qwen2.5:7b

python -m mictotext
```

Vedi `--help` per tutte le opzioni:

```powershell
python -m mictotext --help
```

## Interfaccia web (opzionale)

Oltre al flusso da riga di comando, c'è una piccola interfaccia grafica che
si apre nel browser, utile se preferisci un pulsante a `premi INVIO`:

```powershell
python -m mictotext.web
```

Apre automaticamente `http://127.0.0.1:8765/` (solo su questo PC, non
esposto in rete). La pagina permette di scegliere microfono/lingua/modelli,
avviare e fermare la registrazione con un pulsante, seguire il log
dell'elaborazione in tempo reale e vedere appunti e schema al termine, con
link per scaricare `appunti.md`, `schema.mmd` e `trascrizione.txt`.

Il microfono resta comunque catturato dal processo Python sul PC (tramite
`sounddevice`, come nel flusso CLI): il browser è solo il telecomando e il
visualizzatore dei risultati, l'audio non transita mai sulla rete. È uno
strumento per un solo utente/una sessione alla volta, non un server
multiutente: se una registrazione o un'elaborazione sono in corso, avviarne
un'altra viene rifiutato finché la prima non finisce.
