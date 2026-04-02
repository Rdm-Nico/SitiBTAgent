# Environment Variables 
Questa é una piccola sezione che spiega quali sono le variabili d'ambiente utilizzate nella app e per certe di queste, definisce come ottenerle. 



| Variabile | Utilizzo | Locazione | Descrizione |
|---|---|---|---|
| **SQLSERVER_PASSWORD** | [config.yaml](/config.yaml) | `/home/tecnico/.bashrc`  | Password dell'utente che si collega a sql server
| **POSTGRES_PASSWORD** | [config.yaml](/config.yaml) | `/home/tecnico/.bashrc` | Password dell'utente che si collega a postgres sql |
| **ACCESS_TOKEN_TECNICO** | [config.yaml](/config.yaml) | `/home/tecnico/.bashrc` | Access token per l'utente di sistema della app Meta*|
| **VERIFY_TOKEN_TECNICO** | [config.yaml](/config.yaml) | `/home/tecnico/.bashrc` |  verify token per l'utente di sistema della app Meta*|
| **PHONE_ID** | [config.yaml](/config.yaml) | `/home/tecnico/.bashrc`  | phone id  della app Meta|
| **NGROK_AUTHTOKEN** | [podman-compose.yml](/production/podman/podman-compose.yml) | `/home/tecnico/.bashrc`  | Access token di ngrok|
| **HF_TOKEN** | [vllm-installer.sh](/production/podman/vllm-installer.sh) | `.bashrc` nella home del server GPU  | Access token di Hugging Face|
---

 *Queste variabili d'ambiente sono duplicate perché ogni app. es: `ACCESS_TOKEN_TECNICO`, `ACCESS_TOKEN_FOLLOW_UP`,ecc...


## Whatsapp  token accesso
Per quanto riguarda i token di accesso per le due app che utilizzano i servizi Meta, questi token non hanno scadenza prestabilitá, se si dovesse trovare necessario cambiarli, seguire questa guida [qui](https://developers.facebook.com/documentation/business-messaging/whatsapp/access-tokens).

## Hugging Face token accesso
Per certi modelli, come per esempio `google/embeddinggemma-300m`, questi non sono resi disponibile al pubblico direttamente, ma é necessario creare un access token specifico per quella modello che vuoi utilizzare, per fare ció  é molto semplice:
1. Seguire questa guida per creare un access token su [HF](https://huggingface.co/docs/hub/security-tokens);
2. trova la sezione *Repositories permissions* e copiare il nome della repo(es: `google/embeddinggemma-300m`);
3. con l'utente loggato su hugging Face andare alla pagina della repo e acconsetire l'accesso alla repo. 






