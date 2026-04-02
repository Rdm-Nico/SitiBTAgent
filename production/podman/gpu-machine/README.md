# GPU vLLM Setup
Questa  guida mostra come viene avviato vLLM in una macchina.
Viene utilizzato podman-compose perchè è risultato il metodo più pulito e rapido per gestire il framework. 
## Prerequisiti
Controllare che nella macchina sia presente `podman`  e che `podman` abbia il container access alla gpu. 

## Procedura
È necessario copiare i files `vllm-installer.sh` e `otel-agent-config.yaml` nel relativa macchina collegata alla rete aziendale.
*  `vllm-installer.sh`: file per avviare il podman pod, questo file avvia i seguenti containers:
   *  Il modello AI che fa tool calling;
   *  Il modello AI che crea gli embedding;
   *  programma openlit che invia la telemetria della gpu all'endpoint `$COLLECTOR_ENDPOINT`;
   *  un sidecar per catturre la telemetria di vllm e inviarla all'endpoint `$COLLECTOR_ENDPOINT`;
  

* `otel-agent-config.yaml`: file di configurazione per le telemetrie vllm;


Successivamente è semplicemente necessario avvia il file bash:
```bash
./vllm-installer.sh
``` 