# Modelli per ogni step della pipeline — ricerca aggiornata

**Data ricerca:** 18 settembre 2026
**Hardware di riferimento:** ASUS TUF — NVIDIA RTX 5060 (8 GB VRAM), Ryzen 7 260, 16 GB RAM
**Progetto:** MicToText

---

## Premessa: 2 dei 5 step non sono problemi da modello

Questa è la conclusione più importante della ricerca, perché fa risparmiare tempo e VRAM:

- **Registrazione** → è cattura audio, non inferenza. Nessun modello.
- **Impaginazione e gestione spaziale degli schemi** → è un **motore di layout** (algoritmo
  deterministico), non un LLM. Nessun modello, locale o cloud, dispone i nodi meglio di un
  layout engine dedicato.

L'LLM serve davvero solo in 2 step: pulizia dei concetti e generazione della sintassi Mermaid.

---

## Step 1 — Registrazione

**Nessun modello necessario.** `sounddevice` fa già tutto.

L'unica aggiunta teoricamente sensata sarebbe un **denoiser** (DeepFilterNet 3, gira su CPU),
ma **la ricerca dà evidenze contraddittorie**:

- un caso documentato riporta un *peggioramento* del ~20% di WER (il denoiser taglia rumore
  ma anche frequenze utili);
- altra ricerca riporta miglioramenti del 20–40% relativo su audio realmente rumoroso.

**Verdetto: non aggiungerlo.** Il VAD Silero già usato da `faster-whisper` (`vad_filter=True`)
è l'intervento utile, ed è già attivo nel progetto. Se un giorno registri in ambiente rumoroso,
testalo *sul tuo audio* prima di adottarlo, non a scatola chiusa.

---

## Step 2 — Trascrizione

| Modello | Dove gira | Italiano | Note |
|---|---|---|---|
| **faster-whisper `large-v3-turbo`** (attuale) | GPU 8GB (~2GB) | Ottimo | Scelta attuale, tuttora valida |
| **NVIDIA Parakeet TDT 0.6B v3** | GPU e ottimo su CPU | ~4,25% WER riportato | **Il vero upgrade**: batte Whisper large-v3 a 1/4 della dimensione, molto più veloce, copre 25 lingue europee incl. italiano |
| Whisper `large-v3` completo | Sconsigliato | Ottimo | Più lento e più VRAM di turbo, qualità quasi identica: spreco |
| Moonshine | Leggerissimo | Solo inglese | Non adatto |

**Nota pratica su Parakeet per Windows:** NON installarlo via NeMo (tira dentro torch, pesante
e lento su CPU). Usare il pacchetto **`onnx-asr`** (richiede ONNX Runtime ≥ 1.25), che carica il
modello da Hugging Face senza torch. È l'unico approccio sensato su questo setup.

> **Raccomandazione:** tenere `large-v3-turbo` come default, ma vale la pena testare Parakeet v3
> — potrebbe essere *insieme* più veloce e più accurato sull'italiano. È l'unico step dove un
> cambio di modello può dare un guadagno doppio.

**Quali NON in locale:** nessuno. La trascrizione è lo step più adatto al locale in assoluto.

---

## Step 3 — Normalizzazione e pulizia dei concetti

`qwen2.5:7b` (scelta iniziale del progetto) è ormai datato. Il panorama si è mosso:

| Modello (tag Ollama) | Dim. | Sta negli 8GB? | Per cosa |
|---|---|---|---|
| **`qwen3.5:9b`** | 6.6 GB | Sì, interamente | **Miglior scelta per 8GB.** ~55–58 t/s misurati su RTX 3070 8GB (comparabile alla 5060), contesto nativo 262K |
| `qwen3.5:4b` | 3.4 GB | Sì, con molto margine | Se serve velocità e VRAM libera |
| `gemma4:e4b` | ~4 GB | Sì | Alternativa multilingue (140+ lingue), Apache 2.0 |
| `ministral-3:8b` | ~5 GB | Sì | Mistral è forte sulle lingue europee |
| `qwen2.5:7b` (attuale) | 4.7 GB | Sì | Funziona, ma superato dalla generazione 3.5 |

**Sull'italiano:** le famiglie Qwen (multilingue nativo, 100+ lingue) e Mistral (forte sulle
lingue europee) sono le più indicate.

**Evitare i fine-tune italiani specifici** tipo DanteLLM o Anita: sono basati su Mistral 7B /
Llama 3 ormai vecchi, quasi certamente peggiori di un Qwen3.5 generalista aggiornato.

**Quali NON in locale:** `qwen3.5:27b` (17 GB), `:35b` (24 GB), `:122b` (81 GB). Attenzione anche
ai tag `-mlx`: sono per Apple Silicon e comunque `9b-mlx` pesa 8.9 GB, fuori portata.

---

## Step 4 — Generazione sintassi Mermaid

Dato di ricerca rilevante: il benchmark accademico **MermaidSeqBench** (NeurIPS) ha testato
proprio modelli di questa fascia — Qwen 2.5 (0.5B e 7B), Llama 3.1/3.2 (1B e 8B), Granite 3.3
(2B e 8B) — e conclude che ci sono **"significativi divari di capacità"**: i modelli piccoli
sbagliano spesso la sintassi.

| Opzione | Valutazione |
|---|---|
| **Stesso modello degli appunti (`qwen3.5:9b`)** | **Consigliato.** Zero swap di VRAM, il modello è già caricato |
| `qwen2.5-coder:7b` | Migliore sulla sintassi pura, ma su 8GB Ollama deve scaricare un modello e caricare l'altro ad ogni schema |

**Punto chiave:** con 8 GB, il **ciclo di validazione e correzione** già implementato nel progetto
(`mmdc` restituisce l'errore di parsing → l'LLM corregge) conta **più della scelta del modello**.
È esattamente la contromisura al problema che il benchmark documenta.

**Cosa NON provare in locale:** il benchmark usa come giudici DeepSeek-V3 (671B) e GPT-OSS (120B).
Un'eventuale valutazione automatica della qualità degli schemi (LLM-as-judge) è fuori portata per
8 GB: non ha senso tentarla in locale.

---

## Step 5 — Impaginazione e gestione grafica/spaziale

**Non è un lavoro da LLM**, ed è la scoperta più utile della ricerca. Mermaid ha sostituito il
vecchio motore **Dagre** con **ELK** (Eclipse Layout Kernel), che fa routing ortogonale e riduce
le sovrapposizioni.

### Verifica fatta sul setup reale (mermaid-cli 11.16.0)

Stesso identico schema renderizzato nei due modi:

| | Dagre (default attuale) | ELK |
|---|---|---|
| Larghezza | 1540 px, molto orizzontale | **740 px, compatto e verticale** |
| Archi | Curvi, si incrociano a metà canvas | **Ortogonali, aggirano invece di incrociare** |
| Gruppi | Affiancati, molto spazio sprecato | **Impilati in modo pulito** |
| Uso negli appunti | Scomodo (troppo largo) | **Adatto a un documento verticale** |

Si attiva con **una riga di configurazione, nessun modello**:

```
---
config:
  layout: elk
---
flowchart TD
  ...
```

### Divisione dei compiti corretta

- **LLM** → la *semantica*: cosa raggruppare con cosa, quali categorie, quali relazioni.
- **Layout engine** → la *geometria*: dove mettere i nodi, come instradare gli archi.

Chiedere a un modello 9B di fare disposizione spaziale significa sprecare token per un risultato
peggiore.

---

## Budget VRAM sugli 8 GB

La pipeline è già sequenziale, quindi i picchi non si sommano:

| Fase | Occupazione | Note |
|---|---|---|
| Whisper turbo (processo figlio) | ~2 GB | Liberati completamente all'uscita del processo |
| `qwen3.5:9b` Q4 + contesto 16K | ~6,6 GB + ~1 GB KV | Ci sta, ma è il limite |
| Rendering Mermaid | ~0 GB GPU | Chromium headless: usa CPU/RAM |

Con 16 GB di RAM il margine è sufficiente. In caso di swapping: scendere a `qwen3.5:4b` oppure
ridurre `--num-ctx` a 8192.

---

## Interventi consigliati, in ordine di guadagno

1. **ELK come layout di default** — costo: una riga di configurazione. Guadagno: schemi nettamente
   più leggibili. Verificato funzionante su `mmdc` 11.16.0 di questa macchina.
2. **`qwen2.5:7b` → `qwen3.5:9b`** — costo: un `ollama pull` (6,6 GB). Guadagno: qualità appunti e
   sintassi Mermaid superiori, stessa VRAM.
3. **Testare Parakeet TDT v3 via `onnx-asr`** — costo: lavoro di integrazione. Guadagno potenziale:
   più veloce *e* più accurato sull'italiano.
4. **Non aggiungere denoising** — l'evidenza disponibile non lo giustifica.

---

## Fonti

- [Gladia — Best open-source speech-to-text models in 2026](https://www.gladia.io/blog/best-open-source-speech-to-text-models)
- [Northflank — Best open source STT model in 2026 (with benchmarks)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [OpenWhispr — Parakeet vs Whisper vs Nemotron: Best Local STT 2026](https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron)
- [onnx-asr — PyPI](https://pypi.org/project/onnx-asr/)
- [LocalLLM.in — Best Local LLMs for 8GB VRAM: Real Hardware Benchmarks](https://localllm.in/blog/best-local-llms-8gb-vram-2025)
- [InsiderLLM — Qwen 3.5 9B setup guide for 8GB GPUs](https://insiderllm.com/guides/qwen-3-5-9b-setup-guide/)
- [Ollama library — qwen3.5 tags](https://ollama.com/library/qwen3.5)
- [MermaidSeqBench: An Evaluation Benchmark for LLM-to-Mermaid Generation (arXiv)](https://arxiv.org/html/2511.14967)
- [Mermaid — Layout engines documentation](https://mermaid.ai/open-source/config/layouts.html)
- [draw.io — Set the Mermaid layout engine (ELK)](https://www.drawio.com/docs/manual/mermaid/mermaid-layout-engine/)
- [DeepFilterNet issue #483 — degrades quality for STT](https://github.com/Rikorose/DeepFilterNet/issues/483)
- [Forasoft — Speech Recognition Accuracy in Noise: 2026 Playbook](https://www.forasoft.com/blog/article/speech-recognition-accuracy-noisy-environments)
- [SiliconFlow — Best Open Source LLM For Italian in 2026](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Italian)
