# Overview

## Folder view
```bash
.
├── docs
│   ├── arch.md
│   └── setup.md
├── models
│   ├── modelfiles
│   │   ├── chat_followup_modelfile.json
│   │   ├── Chat_Modelfile
│   │   ├── chat_modelfile.json
│   │   ├── extractor_followup_modelfile.json
│   │   ├── Extractor_Modelfile
│   │   ├── extractor_modelfile.json
│   │   ├── __init__.py
│   │   └── translate_modelfile.json
│   ├── __pycache__
│   ├── __init__.py
│   ├── ModelProviderClient.py
│   ├── OllamaHttpClient.py
│   └── system_prompts.txt
├── production
│   ├── podman
│   │   ├── postgres-podman
│   │   ├── Containerfile_follow_up
│   │   ├── Containerfile_tecnico
│   │   ├── crontab.sh
│   │   ├── monitor.sh
│   │   ├── ngrok.yml
│   │   ├── obs_compose.yml
│   │   ├── otel-collector-config.yml
│   │   ├── podman-compose.yml
│   │   ├── prometheus.yml
│   │   ├── README.md
│   │   ├── statsd.conf
│   │   └── vllm-installer.sh
│   ├── __init__.py
│   └── scheduler_service.py
├── schema
│   ├── __pycache__
│   ├── db.py
│   ├── __init__.py
│   ├── postgres_models.py
│   └── structured_ouput_model.py
├── training_Siti_agent_use_case
│   ├── __pycache__
│   ├── Containerfile
│   ├── runner_isolation.py
│   ├── siti_agent.py
│   └── train_siti_agent.py
├── utils
│   ├── __pycache__
│   ├── configurator.py
│   ├── Event.py
│   ├── __init__.py
│   ├── logger.py
│   ├── setup_check.py
│   ├── Tool.py
│   └── util.py
├── config_follow_up.yaml
├── config_tecnico.yaml
├── config.yaml
├── README.md
├── requirements.txt
├── run_production.py
├── run.py
├── whatsapp_agent_follow_up.py
└── whatsapp_agent_tecnico.py
```

# Struttura dell'applicazione Tecnico

L'intera struttura dell'applicazione può essere rappresentata tramite il diagramma a package mostrato in [figura](#package-view).

`Run` è il punto di ingresso dell'esecuzione ed è responsabile dell'interrogazione iniziale al database dei tecnici. Tutte le impostazioni di configurazione — come le credenziali del database, le credenziali di WhatsApp e altre — vengono caricate da un file YAML passato come input a `Run`, una spiegazione dei campi presenti in questo file lo poi trovare [qui](/docs/setup.md#file-di-configurazione). Queste impostazioni vengono validate e gestite dalla classe `Configurator`, che centralizza l'accesso a tutte le informazioni di configurazione. Il modulo `Run` avvia sia uno `Scheduler` che un `Gunicorn Runner`, ognuno dei quali crea un'istanza della classe `Whatsapp_Bot`, con ruoli differenti.

---

## Istanza Scheduler

L'istanza di `Whatsapp_Bot` creata dallo Scheduler stabilisce la connessione con il database dei tecnici per recuperare le informazioni sugli utenti, istanziando la classe `DB_Commesse`. È inoltre responsabile dell'invio dei messaggi template iniziali e della creazione delle tabelle necessarie in un database PostgreSQL, che svolgono due scopi principali:

- **Memorizzazione dei messaggi:** salvataggio dei messaggi scambiati tra il bot e gli utenti, incluse le tool call dell'agente AI e i relativi risultati. Questo compito è gestito dalla sottoclasse `DB_Messaggi`, che si appoggia al modello Pydantic `MessaggioTecnico` per definire lo schema della tabella e garantire l'integrità dei dati.
- **Gestione del vector database:** gestione del database vettoriale utilizzato durante la fase di classificazione. Questo compito è assegnato alla sottoclasse `DB_Vector`, che utilizza il modello Pydantic `Etichetta`.

---

## Istanza Gunicorn Runner

Poco dopo l'avvio dello Scheduler, il `Gunicorn Runner` crea una seconda istanza di `Whatsapp_Bot`. Questa funge da **orchestratore principale** e sfrutta tutti i componenti core:

| Componente | Ruolo |
|---|---|
| `Inference Client` | Genera le risposte del bot |
| `DB_Messaggi` | Recupera i messaggi esistenti per il contesto e salva i nuovi |
| `DB_Vector` | Ricerca embedding simili durante la classificazione |
| `DB_Commesse` | Aggiorna i record dei tecnici a fine conversazione tramite stored procedure |

Quando un evento arriva al webhook, può essere di vari tipi:

- **Aggiornamento di stato** — `read`, `sent`, `failure`, `delivered` — per i messaggi inviati dal bot
- **Messaggio in arrivo** dall'utente

`Whatsapp_Bot` incapsula queste informazioni nella classe `Event`, che esegue un'elaborazione iniziale ed espone i dati rilevanti come attributi di classe:

```
message_type | message_id | message_body | sender_id | audio_id | ...
```

---

## Inference Client e attori LLM

Un'altra responsabilità fondamentale di `Whatsapp_Bot` è istanziare il corretto **Inference Provider** per effettuare le richieste al modello. Se il file di configurazione YAML specifica l'utilizzo di un motore di inferenza, il bot crea **quattro istanze di `Inference Client`**, ciascuna della sottoclasse specifica per quel motore.

> Sono necessari quattro client separati perché ognuno corrisponde a un distinto attore LLM, il che semplifica la gestione dei dati personali e delle configurazioni specifiche per modello.

| Attore | Classe | Dipendenze aggiuntive |
|---|---|---|
| **AI Agent** | `Inference Client` + `Tool` | `PushTool`, `ExtractorTool` |
| **Extractor LLM** | `Inference Client` | Pydantic model `Tecnico` (JSON output) |
| **Translation LLM** | `Inference Client` | — |
| **Embedding Model** | `Inference Client` | — |

### AI Agent

L'**AI Agent** estende `Inference Client` sfruttando la superclasse `Tool`. Le due sottoclassi concrete definiscono il tipo di tool, il nome, la descrizione e i parametri in un formato compatibile con il motore di inferenza:

- `PushTool`
- `ExtractorTool`

### Extractor LLM

L'**Extractor LLM** utilizza il modello Pydantic `Tecnico` per generare output strutturato in formato JSON.

### Translation LLM ed Embedding Model

I restanti due componenti non richiedono dipendenze aggiuntive.
## Package View
![package diagram](/docs/imgs/FlowChart%20simile%20Package%20Digram-2026-03-09-144021.png)

---
## Sequence Diagram 

Una vista parziale di come si svolge uno scambio tra un utene e il bot è il seguente:
![sequence diagram](/docs/imgs/Sequence_Diagram_partial-2026-03-09-144038.png) 
---

Una vista parziale di come funziona il pre processing di un messaggio è il seguente:
![sequence diagram](/docs/imgs/Sequence%20Diagram%20Pre%20process-2026-03-09-144100.png)

---
## Deploy Diagram
![deploy diagram](/docs/imgs/API-Driven%20Database%20Storage-2026-03-09-145824.png)

Il deployment segue un classico approccio orientato ai servizi con componenti containerizzati, come illustrato in figura. All'interno della VM, un orchestratore `Podman Compose` gestisce quattro servizi:

- **WhatsApp App Service:** contiene tutta la logica applicativa descritta nelle sezioni precedenti
- **Ngrok Service:** stabilisce un tunnel Ngrok per esporre l'endpoint webhook richiesto dalle API di WhatsApp
- **PostgreSQL Service:** ospita sia il database dei messaggi che il vector database. Questo servizio utilizza un *bind mount* per garantire la persistenza dei dati oltre il ciclo di vita del container
- **OpenTelemetry Exporter:** raccoglie ed esporta i dati di telemetria per il monitoraggio dell'applicazione

---

## Connessioni esterne

Dall'ambiente Compose dell'applicazione si diramano quattro connessioni esterne.

### WhatsApp APIs

L'invio e la ricezione di messaggi tramite WhatsApp richiede un account Meta Business e un'applicazione associata che esponga gli endpoint API necessari.

### SQL Server

Questa connessione collega la WhatsApp App al database dei tecnici per il recupero e l'aggiornamento delle informazioni. La connessione viene stabilita tramite un account utente dedicato con privilegi di sola lettura per il recupero dei dati, mentre gli aggiornamenti vengono eseguiti invocando una stored procedure.

### GPU Server

L'azienda ha messo a disposizione due macchine fisiche per l'esecuzione locale dei modelli AI. Eseguire i modelli on-premise, anziché affidarsi a chiamate API esterne, richiede una macchina dotata di GPU e sufficiente memoria video (VRAM), nonché un framework per il model serving.

L'applicazione supporta due motori di inferenza: **vLLM** e **Ollama**.

| | Ollama | vLLM |
|---|---|---|
| **Utilizzo** | Testing iniziale | Produzione |
| **Setup** | Semplice | Curva di apprendimento più ripida |
| **Performance** | Limitazioni di efficienza e latenza | Ottimizzato per produzione |
| **Ottimizzazioni** | — | Paged Attention, Chunked Prefill, Parallel Deployment |


### Monitoring Server

Un server dedicato ospita un secondo ambiente `Podman Compose` responsabile dell'aggregazione di tutti i dati di telemetria dai servizi precedentemente descritti. Include un'istanza **Grafana** per la visualizzazione delle metriche raccolte, rendendo i dati di telemetria dell'intera applicazione accessibili da un unico endpoint.


# Applicazione Follow Up
L'app sui follow up seguente la stessa struttura del tecnico, con qualche differenza:
 1.   Lo scheduler viene avviato ogni lunedì per inviare ai commerciali in ritardo con l'aggiornamento dell'offerta;
 2.   Il gunicorn server è attivo per tutta la settimana aspettando le risposte;