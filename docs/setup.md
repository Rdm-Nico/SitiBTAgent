# Setup 

Questo file definisce in breve queli siano le best practice prima di avviare personalmente i due applicativi (tecnico e follow up). 


Per una overview di come l'app è strutturata e della sequenza di come vengono eseguite le funzioni fare riferimento alla pagina [arch](/docs/arch.md)


**P.S.** Per quanto riguarda l'ambiente di produzione si utilizza podman-compose, per maggiore informazioni seguire questa [guida](/production/podman/README.md).

## Checklist before running
Controllare che siano sempre validi questi punti prima di far partire l'applicativo:
- [ ]  controllare che l’immagine di whatsapp-agent per ogni app sia la più stabile;
- [ ]  controllare che nel file di produzione ci sia `env: PROD`  e `is_db: True`;
- [ ]  controllare che il server dove è servito vLLM/Ollama sia online;
  

## GPU servers Provider Setup
Al momento della scrittura sono presenti due macchina con una gpu abbastanza potente per fare girare dei modelli AI:
- 172.18.30.178; 
- 172.18.31.238;
  
### Ollama setup
Per quanto riguarda l'utilizzo di Ollama bisogna controllare che sia installato e aggiornato correttamente nella macchina e i campi `ollama.*_model` del file di configurazione corrispondono ai modelli disponibili in locale nel server GPU in cui Ollama sta girando.

### vLLM setup
vLLM invece utilizza un file bash che avvia un podman pod. Questo file  è presente direttamente nella  macchina con la GPU è *deve* essere avviato ogni giorno(il processo è automaticamete gestito da crontab). Maggiore informazioni per avviare il pod li puoi leggere [qui](/production/podman/gpu-machine/README.md).

## File di Configurazione
Per eseguire una delle app a dispozione bisogna fare riferimento al file di configurazione `config.yaml`. Un file di configurazione è sviluppato in questo modo:
```yaml
app:
  # che app avviare 
  name: tecnico
server:
  # server a cui fare riferimento per fare partire il modello AI
  url: 172.18.31.178
provider: 
  # provider da utilizzare per fare partire il modello AI
  use: vllm
vllm:
  # vllm provider, usare questo per produzione 
  # iperparametri e system prompt per ogni modello AI da utilizzare 
  chat_model:
    source: chat-tuned
    url: http://${server.url}:8000
    name: chat_agent
    path: ./models/modelfiles/chat_modelfile.json
  extractor_model:
    source: unsloth/Qwen3-4B-Instruct-2507-bnb-4bit
    url: http://${server.url}:8000
    name: extractor_expert
    path: ./models/modelfiles/extractor_modelfile.json
  translate_model:
    source: unsloth/granite-4.0-micro-unsloth-bnb-4bit
    url: http://${server.url}:8000
    name: translate_expert
    path: ./models/modelfiles/translate_modelfile.json
  embedding_model:
    source: google/embeddinggemma-300m
    url: http://${server.url}:8001
ollama:
  # ollama provider, usare questo per fase di sperimentazione dei modelli
  url: http://${server.url}:11434
  chat_model:
    source: granite4:latest
    name: chat_agent
    path: ./models/modelfiles/chat_modelfile.json
  extractor_model:
    source: granite4:latest
    name: extractor_expert
    path: ./models/modelfiles/extractor_modelfile.json
  translate_model:
    source: granite4:latest
    name: translate_expert
    path: ./models/modelfiles/translate_modelfile.json
  embedding_model:
    source: embeddinggemma:latest
  generate_training_model: gpt-oss:20b
database:
  # booleano che fa utilizzare un file csv al posto di chiamare sql server per testing
  is_db: False
  # host che nel caso di podman compose è il nome del servizio
  postgres_server: localhost
  sqlserver_server: srvlayer
  sqlserver_db_name: AI_REPO
  sqlserver_user: ai_repo
  sqlserver_password: ${oc.env:SQLSERVER_PASSWORD}
  sqlserver_conn: DRIVER={ODBC Driver 18 for SQL Server};SERVER=${.sqlserver_server};DATABASE=${.sqlserver_db_name};UID=${.sqlserver_user};PWD=${.sqlserver_password};TrustServerCertificate=yes
  postgres_port: 5432
  postgres_db_name_msg : msgdb
  postgres_db_name_vector : vectordb
  postgres_user: postgres
  postgres_password: ${oc.env:POSTGRES_PASSWORD}
  postgres_msg_conn: host=${.postgres_server} port=${.postgres_port} dbname=${.postgres_db_name_msg} user=${.postgres_user} password=${.postgres_password}
  postgres_vectordb_conn: host=${.postgres_server} port=${.postgres_port} dbname=${.postgres_db_name_vector} user=${.postgres_user} password=${.postgres_password}
  postgres_msg_table: commesse
  postgres_vc_table: etichette
  vector_dim: 512
whatsapp:
  # info whatsapp 
  port: 5000
  access_token:  ${oc.env:ACCESS_TOKEN}
  verify_token: ${oc.env:VERIFY_TOKEN}
  version: v24.0
  phone_id: ${oc.env:PHONE_ID}
  url: https://graph.facebook.com/${.version}/${.phone_id}/messages
whisper:
  # info modello Speech to Text 
  model: small
info:
  # settings generali  
  # ambiente: TEST/PROD
  env: PROD
  # salvare le chat in formato csv dentro alla cartella history
  save_chat: False
  # per salvare i dati in un csv per fare un report dei messaggi inviati
  save_for_report: False
```
## Avviare l'app tecnico in modalità produzione
facendo partire il programma attraverso:
```bash
python run.py -c config.yaml
```

con `info.env = PROD` e `app.name = tecnico`  nel file di configurazione il programma si avvia in due modalità parallelamente:

 * **Scheduler:** avvia il file  `whatsapp_agent_tecnico.py` per fare il fetch dei dati dei tecnici da inviare il messaggio e fa il setup della tabella su Postgers
 * **Executor:** avvia il file `whatsapp_agent_tecnico.py` e crea un server WSGI Gunicorn per ascoltare a wehooks. 
  
  **Suggerimento:** Non chiamare direttamente da command line ma utilizzare il podman compose presente nella cartella [production](/production/podman/).
  


## AI providers
I due providers locali che utilizziamo sono Ollama e vLLM, entrambi utilizzano la stessa logica sottostante cosí da rendere il cambio tra i due il piú velocemente possibile.
### Cambiare AI provider
Per cambiare il provider AI basta selezionare nel file di configurazione il campo  `provider.use` e controllare che le seguenti condizioni sono presenti:

Per vLLM:
- [ ] il pod vLLM sia in esecuzione all'indirizzo indicato da `server.url` con il modello indicato da `vllm.*.source`;
- [ ] il file json specificato per ogni ruolo sia presente nel path indicato da `vllm.*.path`;

Per Ollama:
  - [ ] Ollama è in esecuzione all'indirizzo indicato da `server.url`;

### Cambiare modello AI
Per utilizzare un altro modello al posto di quelli giá presenti si puó seguire questa checklist:
#### Per vLLM:
  1. andare nel rispettivo server GPU e nel file [vllm-installer.sh](/production/podman/gpu-machine/vllm-installer.sh) presente modificare il campo `--model`;
  2. andare nel file di configurazione specifico della app e modificare i/il campi/o `vllm.*.source` con il nome specifico;
  3. avviare l'applicazione.

**P.S.**: vLLM fa il download dei modelli da Hugging Face(HF), questo vuole dire che certi modelli sono "bloccati" se prima non ci si autentica tramite HF alla repo specifica. Per maggiori informazioni guardare la pagina riguardante le [variabili d'ambiente](/docs/vars.md).


---

#### Per Ollama:
  1. andare nel file di configurazione specifico della app e modificare i/il campi/o `ollama.*.source` con il modello specifico;
  2. avviare l'app. 

### Cambiare System prompt o configurazioni
Per cambiare il system prompt che il modello legge oppure altri iperparametri (`top-p`, `temperature`, ecc...) bisogna andare nella cartella `models/modelfiles/` e modificare i file json specifici per ogni ruolo della applicazione.

## Avviare l'app follow-up in modalità test
facendo partire il programma attraverso:
```bash
python run.py -c config.yaml
```

con `info.env` = TEST e `app.name` = follow-up  nel file di configurazione il programma si avvia in una modalità:

 * **Flask App:** avvia il file  `whatsapp_agent_follow_up.py`  come un app Flask e svolge tutto il lavoro lui;
  
  **Suggerimento:** quando si lavora in modalità di test settare per sicurezza anche il campo `database.is_db` pari a `true` per fare in modo che il fetch dei dati sia fatto da un file csv locale e anche gli update vengono svolti sullo stesso file locale.  
