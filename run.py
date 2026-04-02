from whatsapp_agent_tecnico import Whatsapp_Agent as WA_tecnico
from whatsapp_agent_follow_up import Whatsapp_Agent as WA_follow
from utils.configurator import Configurator
from  utils.logger import Logger
import argparse
import os
import time
from schema.db import DB_Commesse, DB_Followup
import subprocess
import multiprocessing
import sys
import pandas as pd
from datetime import datetime
logger = Logger(save=True, fileLevel="DEBUG").getLogger()


def save_data(db: pd.DataFrame,app:str):
    """funzione per salvare i contatti delle persone a cui arriva il messaggio"""
    
    # aggiungiamo la data di oggi come reference
    db['Messaggio_Inviato_Il'] = datetime.now()

    filepath = f"/app/history/save_data_for_report/{app}/data.csv"
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        data = pd.concat([data, db], ignore_index=True)
    else:
        data = db.copy()

    data.to_csv(filepath, index_label=False, index=False)

def check_db_has_data(config):
    """Funzione che controlla se sono presenti dati dentro al db"""
    # controlliamo che app dobbiamo far partire
    db = DB_Commesse(config=config) if config.app.name == "tecnico" else DB_Followup(config=config)

    try:
        if db.is_empty():
            logger.info("Nessuna riga da processare. l'app non verrà avviata")
            return False
        
        if config.info.save_for_report:
            save_data(db.data, config.app.name)
        return True
    except Exception as e:
        logger.error(f"Errore nel controllo dei dati: {e}")
        return False
    
def start_scheduler():
    """Funzione per avviare lo scheduler"""
    schedul_cmd = ["python3.13", "production/scheduler_service.py"]

    try:
        process = subprocess.Popen(schedul_cmd)
        logger.info("Scheduler avviato")
        return process
    except Exception as e:
        logger.error(f"Errore nell'avvio dello script dello scheduluer: {e}")
        return None
    

def start_gunicorn(conf):
    """
    Funzione per l'avvio di Gunicorn da file 
    """
    # otteniamo il numero di worker ( di solito: 2-4 x numero di CPU cores)
    try:
        workers = multiprocessing.cpu_count() * 2 + 1
        workers = min(workers,8)
    except:
        workers = 4
    
    # cambiamo a seconda dell'app:
    if conf.app.name == "tecnico":
        # comandi gunicorn
        gunicorn_cmd = [
            "gunicorn",
            "-w", str(workers),
            "-k", "gevent",  # Worker type (gevent per più concorrenza)
            "-b", "0.0.0.0:5000",
            "--timeout", "120",
            "--worker-connections", "1000",
            "--access-logfile", "./logs/app.log",  # Log su stdout
            "--error-logfile", "./logs/app.log",   # Error log su stdout
            "--statsd-host","statsd-exporter:9125", # per esporre le metriche
            "--statsd-prefix","app_tecnici",
            "--log-level", "debug",
            "run_production:app"
        ]
    else:
        gunicorn_cmd = [
            "gunicorn",
            "-w", str(workers),
            "-k", "gevent",  # Worker type (gevent per più concorrenza)
            "-b", "0.0.0.0:6000",
            "--timeout", "120",
            "--worker-connections", "1000",
            "--access-logfile", "./logs/app.log",  # Log su stdout
            "--error-logfile", "./logs/app.log",   # Error log su stdout
            "--statsd-host","statsd-exporter:9125", # per esporre le metriche
            "--statsd-prefix","app_follow_up",
            "--log-level", "debug",
            "run_production:app"
        ]

    
    logger.info(f"avvio Gunicorn con {workers} workers")
    logger.info(f"Gunicorn commands: {' '.join(gunicorn_cmd)}")

    try:
        # avvio gunicorn
        # questo blocca il seguente programma finchè Gunicorn non viene stoppato
        subprocess.run(gunicorn_cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Ricevuto SIGINT, shutdown in corso...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        logger.error(f"Errore nell'avvio di Gunicorn: {e}")
        sys.exit(1)


def main():
    # add the config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file", help="Path to config file", default="config.yaml"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forza l'avvio anche senza dati nella vista"
    )
    args = parser.parse_args()
    configurator = Configurator(file=args.conf_file)

    conf = configurator.get_file()

    # se siamo in produzione avviamo Gunicorn
    if conf.info.env == "PROD":
        has_data = False
        if args.force:
            logger.info("Avvio app anche senza vedere se ci sono dati nella vista")
            has_data = True
        else:
            has_data = check_db_has_data(conf)

        if has_data:
            # esporta le config come variabile d'ambiente per l'app 
            os.environ['APP_CONFIG'] = args.conf_file

            
            
            
            logger.info("=" *50)
            logger.info("Avvio Whatsapp Scheduler")
            logger.info("=" *50)
            start_scheduler()

            # aspettiamo 10 secondi che lo scheduler crea la tabella sul db
            time.sleep(10)
            logger.info("=" *50)
            logger.info("Avvio server Whatsapp Gunicorn")
            logger.info("=" *50)
            start_gunicorn(conf)
    else:
        # in test avviamo l'istanza Flask
        if conf.app.name == "tecnico":
            agent = WA_tecnico(conf)
        elif conf.app.name == "follow-up":
            agent = WA_follow(conf)
        while(agent.is_running()):
            continue

if __name__ == "__main__":
    main()