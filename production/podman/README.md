# Podman Setup

### Installazione Linux
```bash
# Debian/Ubuntu
sudo apt-get install podman podman-compose

# Fedora/RHEL
sudo dnf install podman podman-compose
```

## Building the Container Images
Bisogna creare una immagine per ogni app, per esempio per quanto riguarda l'app tecnico:

```bash
cd ..
cd ..
# bisogna trovarsi nella root della repo
podman build -t sitibt-whatsapp-agent-tecnico:latest -f production/podman/Container_tecnico .
```

## Running with Podman

Il `podman-compose.yml` si può avviare attraverso il comando:

```bash
podman-compose -f podman-compose.yml up -d
```

## Vedere i log

```bash
# Using podman
podman logs -f sitibt-whatsapp-agent-tecnico

# Using podman-compose
podman-compose -f podman-compose.yml logs -f
```
## Volumi

Le seguenti dir sono utilizzate per la persistenza dei dati:
- `./logs:/app/logs:Z`: logs
- `./config.yaml:/app/config.yaml:Z`: file configurazione
- `/var/lib/postgresql/data:Z`: storage postresql 
- `/etc/ngrok.yml:Z`: ngrok Configuration file
- `/etc/statsd.conf:Z`: statsd Configuration file
## Clean Up

```bash
# Stop and remove container
podman stop sitibt-whatsapp-agent-tecnico
podman rm sitibt-whatsapp-agent-tecnico

# Or using podman-compose
podman-compose -f podman-compose.yml down

# or a service of a podman-compose 
podman-compose down sitibt-whatsapp-agent-tecnico

# Remove image
podman rmi sitibt-whatsapp-agent-tecnico:latest

# Clean up unused resources
podman system prune -a
```

## Crontab Scripts
Il file `crontab.sh` è lo script utilizzato da crontab per avviare ogni giorno l'applicativo. Il job è il seguente:
```bash
45 8 * * * /home/tecnico/SitiBT-Agent-AI/production/podman/crontab.sh
```
per visualizzare i log di crontab bisogna eseguire:
```bash
sudo tail -f /var/log/cron
```
il file `monitor.sh` invece viene utilizzato per inviare i messaggi fino ora ricevuti ad il server SQL. IL job é il seguente:
 ```bash
45 18 * * * /home/tecnico/SitiBT-Agent-AI/production/podman/monitor.sh
```

