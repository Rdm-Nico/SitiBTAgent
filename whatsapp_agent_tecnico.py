from flask import Flask, request, jsonify
import requests
from pydub import AudioSegment
from faster_whisper import WhisperModel
#from OllamaHttpClient import OllamaClient
from schema.structured_ouput_model import Tecnico
from utils.util import *
from utils.Tool import ExtractorTool, PushTool
from utils.Event import Event
from omegaconf import DictConfig
from schema.db import DB_Commesse, DB_Messaggi, DB_Vector
import json
import os
import time
from utils.logger import Logger
from threading import Thread
from datetime import datetime, timedelta
import pydantic_core
from pytz import timezone
import pandas as pd
from schema.postgres_models import MessaggioTecnici, message_role
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor as ConcurrentThreadPoolExecutor
from functools import wraps
from models.ModelProviderClient import clientRouter
import signal
import uuid


logger = Logger(save=False).getLogger()
# prendiamo come riferimento il timezone italiano
TIMEZONE = timezone('Europe/Rome')
""" Client Whatsapp """
class Whatsapp_Agent:
    def __init__(self, config:DictConfig, is_scheduler:bool=False):
        self.logger = logger
        self.config = config
        self.app = Flask(__name__)

     
        # add clients
        self.extractor_client = clientRouter(config, name="extractor_model")
        self.chat_client = clientRouter(config, name="chat_model")
        self.translate_client = clientRouter(config, name="translate_model")
        self.embedding_client = clientRouter(config, name="embedding_model")

        # add tools
        self.chat_client.add_tools(tools=[ExtractorTool(), PushTool()])

        # upload DB delle commesse non realizzate
        self.db = DB_Commesse(config)

        # per vedere se è presente un thread (solo in sperimentazione viene utilizzato)
        self.thread = None
        

        # set up Postgres db 
        self.vector_db = DB_Vector(config)
        self.vector_db.add_table(table_name=config.database.postgres_vc_table)

        self.is_scheduler = is_scheduler
        if config.info.env == "PROD":
            # modalità: Scheduler -> invia i template
            #           Executor -> cotrolla le webhooks
            # aggiungi master gunicorn 
            self.gunicorn_master_pid = os.getppid()
            if is_scheduler:
                # add Postgres msg db
                self.message_db = DB_Messaggi(config, app='tecnico')
                # add scheduler
                jobstores = {'default': MemoryJobStore()}
                executors = {'default': ThreadPoolExecutor(20)}
                # add job defaults:
                # coalesce: non esegue più di una volta lo stesso job
                # misfire_grace_time: evitare che un job venga skippato perchè il suo tempo di esecuzione è finito 
                job_defaults = {'coalesce' : False, 'misfire_grace_time': None}
                
                self.scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=TIMEZONE)
                
                # avvio lo scheduler
                self.scheduler.start()
            else:
                # add Postgres msg db
                self.message_db = DB_Messaggi(config, app='tecnico')
                # set up delle routes 
                self._routes()
                # webook Executor per fare un massimo di messaggi contemporaneamente
                self.webhook_executor = ConcurrentThreadPoolExecutor(max_workers=20, thread_name_prefix="webhook_worker")
        else:
            # avviamo il modalità test con un unico processo
            # add Postgres msg db
            self.message_db = DB_Messaggi(config, app='tecnico')
            # add scheduler
            jobstores = {'default': MemoryJobStore()}
            executors = {'default': ThreadPoolExecutor(20)}
            job_defaults = {'coalesce' : False}
            self.scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults)
            # avvio lo scheduler
            self.scheduler.start()
            # set up delle routes 
            self._routes()
            # webook Executor per fare un massimo di messaggi contemporaneamente
            self.webhook_executor = ConcurrentThreadPoolExecutor(max_workers=20, thread_name_prefix="webhook_worker")

        # avviamo l'app
        self.start()

        
    def _run(self):
        """
        Metodo privato per avviare l'app  quando siamo in fase di sperimentazione
        """
        # use_reloader se True usa un processo di reload per fare il restart del server se il codice cambia
        self.app.run(debug=True, host="0.0.0.0", port=self.config.whatsapp.port, use_reloader=False, use_debugger=False) 

   
    def _create_template(self, row)-> dict:
        """
        Crea il template per il primo messaggio.
        Campi del template di ritorno:
        - name: il nome del template
        - language_code: il codice della lingua(en_US)
        - commessa: numero della commessa
        - data: data della commessa
        - cliente: nome del cliente 
        - MATR: matricola tecnico
        - CODICE_INTERVENTO: codice dell'intervento
        """
        # individuiamo la  lingua da utilizzare 
        language, number = process_tel_number(row['TECNICO_MOBILE'])
        
        # find right template
        match language:
            case 'it': template_name = "template_attivita_tecnico"
            case 'en': template_name = "template_activity_technician"
            case 'es': template_name = "template_actividad_tcnica"
            case _: return {'errore': number}

        # aggiungiamo il testo del template da salvare
        try:
            testo =f"""
                            Questa è l AI di SITI B&T Group.  Questo numero ti segnalerà quando non sono state segnate le ore dell intervento di ieri. 
                            Puoi rispondere a questo numero e inserendo le ore scrivendo per esempio ieri ho svolto 6 ore di lavoro  e 1 ora di viaggio. 
                            Puoi anche inserire note sulla giornata.  Puoi  anche inviare un audio al suo posto.
                            Per ulteriore informazioni riferire al reparto CED.
                            Risulta che non siano state inserite le informazioni riguardanti la commessa del giorno *{row['DATA_INTERVENTO']}*. 
                            Comunicare ore di lavoro, ore straordinarie, ore di viaggio e eventuali inefficienze.
                        """ 

            template = {
                'name' : template_name,
                'language_code' : language,
                'commessa' : str(row['COMMESSA']),
                'data' : datetime.combine(row['DATA_INTERVENTO'], datetime.min.time(), tzinfo=TIMEZONE),
                'cliente' : row['DES_CLIENTE'][:20].strip(),
                'matricola_tecnico': row['MATR'],
                'codice_intervento': row['CODICE_INTERVENTO'],
                'testo': testo.strip()
            }
        except Exception as e:
            self.logger.error(f"errore nella creazione del template per il numero {number}: {e}")
            return {'errore': f"{e}"}

        return template 

    def add_job(self, row, when=0):
        """
        Aggiungi un job nello scheduler per inviare i messaggi
        """ 
        template = self._create_template(row)
        if 'errore' in template:
            self.logger.error(f"Errore nella creazione del messaggio di template per il tecnico {row['MATR']}. Il suo numero di telefono non è corretto: {row['TECNICO_MOBILE']}")
            return
        # processa il numero prima per salvarlo nel db
        language,sender =  process_tel_number(row['TECNICO_MOBILE'])
        if language == 'errore':
            self.logger.error(f"Il tecnico {row['MATR']} non ha il numero di telefono")
            return 
        
        self.logger.debug(f"template creato con dati: {template}, matricola tecnico:{row['MATR']}")
        try:
            if when != 0:
                # schedule 
                run_time = datetime.now() + timedelta(hours=when)
                job = self.scheduler.add_job(
                    self.send_message,
                    'date',
                    run_date=run_time,
                    kwargs={'to': sender, 'template': template},
                    id=f"process_{row['CODICE_INTERVENTO']}"
                )
                logger.debug(f'send schedule job {job.id}')
            else:
                # invio subito
                job = self.scheduler.add_job(
                    self.send_message,
                    kwargs={'to': sender, 'template': template},
                    id=f"process_{row['CODICE_INTERVENTO']}"
                )
                logger.debug(f'send job now {job.id}')
            
        except Exception as e:
            self.logger.error(f"Errore invio a {sender}: {str(e)}")

    def _wait_jobs(self):
        """Funzione per aspettare che tutti i messaggi template siano stati inviati. Questo avviene quando siamo in produzione"""
        while True:
            jobs = self.scheduler.get_jobs()

            time.sleep(60) # controlliamo ogni 60 secondi
            if len(jobs) == 0:
                self.logger.info("tutti i messaggi sono stati inviati")
                break


    def start(self) -> bool:
        """
        Metodo per avviare l'app(in fase di sperimenazione) se non è presente un thead già in esecuzione.
        Se siamo in fase di produzione allora usiamo due modalità diverse per scheduler dei messaggi
        """

        if self.config.info.env == "PROD":
            # controlliamo se siamo in modalità scheduler 
            if not self.is_scheduler:
                self.logger.debug("modalità scheduler non attiva per questo worker")
                return True
            self.logger.info("modalità scheduler: preparazione degli invio dei messaggi")

            tecnici = []
            for _,row in self.db.data.iterrows():

                # controlliamo se non è già stato fatto il primo messggio per l'intervento in questione e non ha lo status 'unknown'
                all_messages = self.message_db.find(self.config.database.postgres_msg_table, by="codice_intervento", value=row['CODICE_INTERVENTO'], order_field="cronologia", order_direction="ASC")
                if all_messages.empty == False:
                    # controlliamo se è in fase unknown allora dobbiamo eliminare il messaggio inviato prima e poi inviare quello nuovo
                    if all_messages.iloc[0]['status'] != 'unknown':                        
                        self.logger.debug(f"il primo messaggio per il tecnico: {row['MATR']} e intervento : {row['CODICE_INTERVENTO']} è già stato inviato non lo aggiungiamo allo scheduler")
                        continue

                    # cancelliamo il o i messaggi precedenti
                    have_error = False
                    for _, row_2_delete in all_messages.iterrows():
                        ris = self.message_db.delete(self.config.database.postgres_msg_table, where_field="msg_id", where_value=row_2_delete['msg_id'])
                        if ris == False:
                            # continuiamo a eliminare gli altri messaggi
                            have_error = True 
                    
                    if have_error:
                        # se c'è stato un errore non inviamo più questo messaggio e passiamo ai prossimi
                        self.logger.error(f"errore nella eliminazione dei messaggi passati per l'intervento {row['CODICE_INTERVENTO']}, non si invia più il messaggio")
                        continue

                    

                # controllo se è presente più di un tecnico
                if row['MATR'] in tecnici:
                    # invia il secondo messaggio il pomeriggio -> 14:30 (6 ore dopo)
                    when=6
                    self.logger.info(f'per il  tecnico: {row['MATR']} il secondo messaggio verrà inviato tra {when} ore')
                else:
                    # non trovato nella lista, inseriscilo dentro
                    tecnici.append(row['MATR'])
                    when=0
                
                # aggiungo i jobs nello scheduler
                self.add_job(row, when)
            self.logger.info(f'invio di {len(self.db.data)} messaggi in background')
            # aspettiamo che i job siamo finiti
            self._wait_jobs()
            # stoppiamo l'app dopo aver completato il lavoro
            self.stop()
        else:
            # controlliamo se nel db sono presenti dati da processare
            ris = self.db_isEmpty()

            if not ris:
                if self.thread is None or not self.thread.is_alive():
                    self.thread = Thread(target=self._run)
                    self.thread.daemon = True
                    self.thread.start()
                    self.logger.info(f"App Whatsapp in esecuzione")

                    # aspettiamo che Flask sia up
                    time.sleep(5)
                    tecnici = []
                    for _,row in self.db.data.iterrows():

                        # controllo se è presente più di un tecnico
                        if row['MATR'] in tecnici:
                            # invia il secondo messaggio il pomeriggio -> 14:30 (6 ore dopo)
                            when=1
                            self.logger.info(f'per il  tecnico: {row['MATR']} il secondo messaggio verrà inviato tra {when} ore')
                        else:
                            # non trovato nella lista, inseriscilo dentro
                            tecnici.append(row['MATR'])
                            when=0

                        # aggiungo i jobs nello scheduler
                        self.add_job(row, when)

                    self.logger.info(f'invio di {len(self.db.data)} messaggi in background')

                else:
                    self.logger.info("L'App Whatsapp è gia in esecuzione")
            else:
                self.logger.info("Non sono presenti righe da processare per oggi. L'App whatsapp non si avvierà")
    
    def is_running(self):
        """
        Controlla se l'app Flask è up
        """
        return self.thread is not None
    
    def stop(self):
        """
        Stop dell'app Flask e degli Executor
        """ 
        # aspettiamo un secondo che il messaggio di stop sia inviato nella webhook
        time.sleep(1)
        # salviamo tutti i messaggi attuali se siamo nella fase di test e vogliamo salvare la cronologia delle chat 
        if self.config.info.save_chat and not self.is_scheduler:
            self.message_db.export_to_csv(self.config.database.postgres_msg_table)
        if self.is_scheduler:
            self.logger.info("Stopping Whatsapp Scheduler...")
            self.scheduler.shutdown(wait=True)
        else:
            self.logger.info("Stopping Whatsapp agent...")
            self.webhook_executor.shutdown(wait=True, cancel_futures=False)
        

        if self.thread:
            self.thread = None
        if self.config.info.env == "PROD":
            # stoppiamo il master Gunicorn
            os.kill(self.gunicorn_master_pid, signal.SIGTERM)

        
        if self.is_scheduler:
            self.logger.info("Whatsapp Scheduler stopped")
        else:
            self.logger.info("Whatsapp Agent stopped")


    def db_isEmpty(self)-> bool:
        """
        Controlla se sono presenti righe nel db ed eventualmente ritorna true così da avviare l'app
        """
        return self.db.is_empty()
    
    def check_and_translate(func):
        """Decoratore per tradurre il messaggio in un altra lingua"""
        @wraps(func)
        def wrapper(self, to:str, message:str=None, template:dict=None) -> str:
            if template == None:
                # controlliamo il numero 
                language, _ = process_tel_number(to)
                if language == "errore":
                    logger.error(f"errore nel rilevamento della lingua del numero {to}")
                    return "errore nel numero rilevato"
                
                if language != 'it':
                    logger.debug(f"Lingua diversa da italiano, tradurre il messaggio in {language}")
                    response = self.translate_client.translate_text(text=message, target_language=language)
                    if 'success' != response['status']:
                        logger.error(f"errore nella traduzione del messaggio del numero {to}")
                        return "errore nella traduzione del messaggio"
                    message = response['content']
                
            return func(self, to, message, template)
        return wrapper

    @check_and_translate
    def send_message(self, to:str, message:str=None, template:dict=None, fail_counter:int=0) -> str:
        """
        Inviare messaggio all'utente e aggiornare la cronologia della conversazione 
        Params:
            - to (str): numero del utente da inviare il messaggio
            - message(str): il messaggio da inviare all'utente 
            - template(dict): se bisogna utilizzare il template(in caso di primi messaggi bisogna utilizzarlo)
            - fail_counter(int): numero di volte il msg è già stato inviato e c'è stato un errore
        """
        headers = {
            'Authorization' : f'Bearer {self.config.whatsapp.access_token}',
            'Content-Type' : 'application/json'
        }
        
        try:
            if template:
                # primo messaggio
                data = {
                    'messaging_product': 'whatsapp',
                    'recipient_type' : 'individual',
                    'to' : to,
                    'type' : 'template',
                    'template': {
                        'name': template['name'],
                        'language':{
                            'code' : template['language_code']
                        },
                        'components': [
                            {
                                'type': 'HEADER',
                                'parameters': [
                                    {
                                        'type': 'TEXT',
                                        'parameter_name' : 'num_commessa',
                                        'text': template['commessa']
                                    }
                                ]
                            },
                            {
                                'type':'BODY',
                                'parameters': [
                                    {
                                        'type': 'TEXT',
                                        'parameter_name' : 'data',
                                        'text': template['data'].strftime('%d/%m/%Y')
                                    },
                                    {
                                        'type': 'TEXT',
                                        'parameter_name' : 'nome_cliente',
                                        'text': template['cliente']
                                    },
                                ]
                            }
                        ]
                    }
                }

            else:
                # normale messaggio
                data = {
                    'messaging_product': 'whatsapp',
                    'recipient_type' : 'individual',
                    'to' : to,
                    'type' : 'text',
                    'text' : {'body':message}
                }

        
        
            response = requests.post(self.config.whatsapp.url, headers=headers, json=data)
            data = response.json()
            self.logger.debug(f"questa è la response della request: {data}")
            msg_id = data['messages'][0]['id']

            self.logger.debug(f"la response al messaggio è id: {msg_id}")
            # salviamo il primo messaggio nel db
            if template:
                try:
                    msg = None
                    
                    # convertiamo in messaggio per Tecnico
                    msg = MessaggioTecnici(ruolo="assistant", commessa=template['commessa'], matricola_tecnico=template['matricola_tecnico'], codice_intervento=template['codice_intervento'],
                                data_intervento=template['data'], cel_mittente=self.config.whatsapp.phone_id, cel_destinatario=to, msg_id=msg_id, last_modify=datetime.now(tz=TIMEZONE),
                                cronologia=0, failed_count=fail_counter, testo=template['testo'])
                    
                except pydantic_core._pydantic_core.ValidationError as err:
                    self.logger.error(f"Errore di validazione del template da inviare al tecnico per la commessa {template['commessa']}: {str(err)}")
                    return
                
                ris = self.save2DB(msg)
                if ris == False:
                    return
                self.logger.debug("primo messaggio salvato correttamente")
            
        except Exception as e:
            self.logger.error(f"Errore nella scrittura del messaggio a {to}: {str(e)}")
            return None
        return data

    



    def download_audio(self, msg_recived:Event) -> str|None:
        """
        Download audio from url
        """
        headers = {
            'Authorization' : f'Bearer {self.config.whatsapp.access_token}'
        }
        filename=f"audio_message_{msg_recived.sender}.ogg"
        self.logger.debug(f"prima della request per ricevere il file audio")
        response = requests.get(msg_recived.audio_url, headers=headers)
        self.logger.debug(f"dopo la response per ricevere il file audio")
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filename
        return None

    def getAudioUrl(self, msg_recived:Event) -> str|None:
        """ 
        Prendere l'audio url dal audio id
        """
        api_version = self.config.whatsapp.version
        audio_id = msg_recived.audio_id
        url = f"https://graph.facebook.com/{api_version}/{audio_id}"
        self.logger.debug(f"url per ricevere l'audio url: {url}")
        headers = {
            'Authorization' : f'Bearer {self.config.whatsapp.access_token}'
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        self.logger.debug(f"la response è {data}")
        if data['url'] != None:
            return data['url']
        return None

    def trascribe_audio(self,audio_path:str)-> str:
        """
        S2T with Fast Whisper & pydub(convert .ogg to .wav)
        """
        # convert .ogg in .wav
        try:
            sound = AudioSegment.from_file(audio_path)
            self.logger.debug(f"trovato il file path {audio_path}")
        
            wav_path = audio_path.replace('.ogg', '.wav')
            sound.export(wav_path, format='wav')
            self.logger.debug(f"convertito il file in wav: {wav_path}")
            # get Whispher model
            model = WhisperModel(self.config.whisper.model, device="cpu", compute_type="int8")

            self.logger.debug(f"modello: {self.config.whisper.model} metto in cpu, adesso si fa il transcript")
            #FIXME: se serve si potrebbe mettere la lingua anche qua
            segments, info = model.transcribe(wav_path, beam_size=5)
            self.logger.debug(f"Lingua trovata {info.language} con probabilità {info.language_probability}")

            transcript = ""
            for segment in segments:
                self.logger.debug("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                transcript += segment.text
            
            # puliamo il file temporaneo wav
            if os.path.exists(wav_path):
                os.remove(wav_path)

            return transcript

        except Exception as e:
            self.logger.error(f"Errore nella trascrizione per il file al path {audio_path}: {str(e)}")


    def error_message(self, to:str, message:str="c'è stato un errore interno, riprova a mandare il messaggio tra qualche secondo"):
        """
        Messaggio di errore da inviare all'utente 
        """
        self.send_message(to=to, message=message)
        
    
    def handleAudioMessage(self, msg_recived:Event)->str|None:
        """
        Funzione per gestire i messaggi audio
        """
        # messaggio audio, bisogna fare avere prima l'url per fare il download
        msg_recived.audio_url = self.getAudioUrl(msg_recived)
        self.logger.debug(f"recuperato l'url dell'audio da scaricare: {msg_recived.audio_url}")
        if msg_recived.audio_url is None:
            self.logger.error("non è stato possibile ricavare l'url dell'audio. Rinnova l'access token")
            return 
        audio_path = self.download_audio(msg_recived)
        if audio_path == None:
            message = "Non sono stato ingrado di processare l'audio. Per favore invia un messaggio di testo"
            response = self.send_message(to=msg_recived.sender, message=message) 
            return 
        self.logger.debug(f"il path dell'audio è: {audio_path}")
        # trascribe 
        body = self.trascribe_audio(audio_path)
        if body == None:
            message = "Non sono stato ingrado di processare l'audio. Per favore invia un messaggio di testo"
            response = self.send_message(to=msg_recived.sender, message=message)
            return 
        
        # clean del file temporaneo dell'audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return body

    def useExtractorModel(self, input:str=None) ->Tecnico:
        """
        Utilizza il modello estrattore per comprendere la risposta dell'utente 
        """
        self.logger.debug(f"input da dare al extractor: {input}")
        model_response = self.extractor_client.generate(prompt=input, schema=Tecnico)
        if model_response['status'] != 'success':
            self.logger.error({"error": f"Errore nell utilizzare il modello estrattore: {model_response['status']['error']}"})
            return None
        clean_output = clean_response(model_response['content'])

        try:
            clean_json = json.loads(clean_output)
            predict = Tecnico.model_validate_json(json.dumps(clean_json))
        except Exception as e:
            self.logger.error({"error": f"Errore: {str(e)}\n model_response: {model_response['content']},\n clean_output: {clean_output}"}) 
            return None
        return predict

    def save2DB(self, obj:MessaggioTecnici)->bool:
        """
        Funzione per salvare nel db postgres il messaggio dell'utente o del bot
        """
        
        if obj.testo or obj.note_commessa or obj.inefficency_note:
            # rimuoviamo eventuali ' diversi per non causare problemi quando inseriamo i dati nel db
            
            if obj.testo:
                obj.testo = remove_single_quotes(obj.testo)
                self.logger.debug(f" body senza più i single quote: {obj.testo} ")
            if obj.note_commessa:
                obj.note_commessa = remove_single_quotes(obj.note_commessa)
                self.logger.debug(f" note senza più i single quote: {obj.note_commessa} ")
            if obj.inefficency_note:
                obj.inefficency_note = remove_single_quotes(obj.inefficency_note)
                self.logger.debug(f" note inefficienza senza più i single quote: {obj.inefficency_note} ")


        ris = self.message_db.insert(table_name=self.config.database.postgres_msg_table, row=obj)   
        return ris


    def analyzeMessage(self, commessa_df:dict, body:str, msg_recived:Event)->bool:
        """
        Funzione per analizzare il messaggio dell'utente tramite la propria commessa 
        """

        # convertiamo il dict in Messaggio così da lavorarci meglio
        try:
            commessa_obj = MessaggioTecnici(commessa=commessa_df['commessa'], matricola_tecnico=commessa_df['matricola_tecnico'], data_intervento=commessa_df['data_intervento'],
                        codice_intervento=commessa_df['codice_intervento'], cronologia=commessa_df['cronologia'])
            
        except pydantic_core._pydantic_core.ValidationError as err:
            self.logger.error(f"Errore nella validazione del messaggio da salvare nel db inviato per tecnico {commessa_df['matricola_tecnico']} numero di cronologia: {str(err)}")
        
        # troviamo tutte le righe del dataset
        all_messages = self.message_db.find(self.config.database.postgres_msg_table, by="codice_intervento", value=commessa_obj.codice_intervento, order_field="cronologia", order_direction="ASC")
        if all_messages.empty == True:
            return False
        
        #self.logger.debug(f"ecco i primi 5 messaggi del codice intervento {commessa_obj.codice_intervento}: {all_messages.head()}")

        context = self.BuildContext(all_messages)

        if context == None:
                return False

        # aggiungiamo il messaggio dell'utente 
        context.append({
            "role": "user",
            "content": body
        })

        #self.logger.debug(f"è stato costruito il contesto:\n{context}\n")
        # facciamo partire il modello 
        chat_response = self.chat_client.chat(message=context, useDB=True)
        if chat_response['status'] != 'success':
            self.logger.error(f"errore nel chat agent: {chat_response['status']['error']}")
            return False

        # controlliamo se richiede l'utilizzo di un tools
        if chat_response['tool_calls']:
            function_name = chat_response['tool_calls']['name']
            self.logger.info(f"il chat model ha chiamato un tool: {function_name}")
            # controlliamo quale funzione ha chiamato:
            if function_name == 'extractor_expert':
                self.logger.debug(f"response del modello dentro al extractor expert : {chat_response}")
                arguments = chat_response['tool_calls']['arguments']
                self.logger.debug(f"valore degli argomenti: {arguments}\t tipo diverso da stringa: {type(arguments) != str}")
                if type(arguments) == str:
                    self.logger.error(f"il tipo di arguments è stringa, ma dovrebbe essere dict. Proviamo a fare un altro json loads")
                    arguments = json.loads(arguments)
                    if type(arguments) == str:
                        self.logger.error(f"anche il secondo tentativo è fallitto, mandiamo messaggio di errore all'utente. Valore del campo arguments: {arguments}")  
                        return False 
                predict = self.useExtractorModel(input=arguments['summary'])
                if predict == None:
                    return False
                self.logger.debug(f"il modello a buttato fuori:\n {predict}\n")
                
                if predict.ore_ordinarie == 0 and predict.ore_straordinarie == 0 and predict.ore_viaggio == 0 and predict.durata_inefficienza == 0 and predict.note == None and predict.commessa == None and predict.note_inefficienza == None:
                    
                    # nessun campo trovato, richiediamo di mandare il messaggio
                    self.logger.debug("il modello non ha trovato nessun campo, richiediamo di riscriverlo")
                    message = "Non ho capito quello che mi hai detto, riscrivimelo"
                    response = self.send_message(to=msg_recived.sender, message=message)
                    # NON salviamo il messaggio nel db per mantenere la coerenza 
                    return True
                
                # 1. possiamo inserire la riga dell'utente con il body e i dati trovati dall'extractor
                commessa_obj.msg_id = msg_recived.msg_id
                commessa_obj.ruolo = message_role.USER
                commessa_obj.cel_mittente = msg_recived.sender
                commessa_obj.cel_destinatario = self.config.whatsapp.phone_id
                commessa_obj.testo = body
                commessa_obj.last_modify = msg_recived.send_time
                commessa_obj.ore_ordinarie = predict.ore_ordinarie
                commessa_obj.ore_straordinarie = predict.ore_straordinarie
                commessa_obj.ore_viaggio = predict.ore_viaggio
                commessa_obj.note_commessa = predict.note
                if predict.durata_inefficienza != 0 and (predict.note_inefficienza != "" and predict.note_inefficienza != "null" and predict.note_inefficienza != "NULL"):
                    # aggiungiamo l'inefficienza
                    self.logger.info("è stata trovata un inefficienza. La si prova a classificare")
                    commessa_obj.find_inefficienza = True
                    commessa_obj.inefficency = predict.durata_inefficienza
                    commessa_obj.inefficency_note = predict.note_inefficienza
                    # chiamiamo la select nel vector db
                    query_emb = self.embedding_client.embed(predict.note_inefficienza, self.config.database.vector_dim)
                    #self.logger.debug(f"risultato della query embeddings: {query_emb}")
                    if query_emb['status'] != 'success':
                        self.logger.error(f"Errore nella realizzare la query di embedding per la commessa {commessa_obj.commessa}: {query_emb['status']['error']}")
                        return False
                    embedding = query_emb['embeddings'][0]['embedding']
                    self.logger.debug(f"query embedding creato con dim {self.config.database.vector_dim}")
                    # misuriamo per distanza euclidea
                    search_result = self.vector_db.find_similar(table_name=self.config.database.postgres_vc_table, query_emb=embedding, method='l2')
                    self.logger.info(f"risulati trovati per la query: {search_result}")
                    # prendiamo il primo elemento come etichetta 
                    #FIXME: qua poi dovremmo utilizzare una politica migliore per prendere l'etichetta.
                    # politica: somma pesata in base alla distanza e al numero di etichette comuni nel top 5
                    type_count = {}

                    for idx in range(len(search_result)):
                        label = search_result.loc[idx, "etichetta"]
                        distanza =  1 - search_result.loc[idx, "distance"]
                        if type_count.get(label):
                            type_count[label] += 1 * distanza
                        else:
                            type_count[label] = 1 * distanza
                    
                    max_etichetta = None
                    max_dist = 0
                    for etichetta,v in type_count.items():
                        if max_dist < v:
                            max_dist = v
                            max_etichetta = etichetta

                    """ etichetta = search_result['etichetta'].iloc[0]
                    distanza = search_result['distance'].iloc[0] """

                    self.logger.info("L'etichetta più simile è {} con distanza: {:.2f}".format(max_etichetta, max_dist))

                    commessa_obj.inefficency_type = max_etichetta
                else:
                    # settiamo a false
                    commessa_obj.find_inefficienza = False

                commessa_obj.status = "read"

                ris_utente = self.save2DB(commessa_obj)
                
                if ris_utente == False:
                        return False
                
                # 2.inviare il messaggio all'utente con i campi trovati
                if commessa_obj.find_inefficienza:
                    message_2_send = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{commessa_obj.find_inefficienza}*\ndurata inefficienza: *{predict.durata_inefficienza}*\netichetta inefficienza: *{commessa_obj.inefficency_type}*\ndescrizione inefficienza: *{predict.note_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{commessa_obj.commessa}*\n\nsono corretti? Se si invia un messaggio tipo 'puoi inviare il messaggio' o 'si va bene'"
                else:
                    message_2_send = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{commessa_obj.find_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{commessa_obj.commessa}*\n\nsono corretti? Se si invia un messaggio tipo 'puoi inviare il messaggio' o 'si va bene' "
                
                response = self.send_message(to=msg_recived.sender, message=message_2_send)
                if response == None:
                    return False
                
                msg_id = response['messages'][0]['id']
                 
                # 3.creare una riga con la chiamata del tool del chat model 
                # incrementimo la cronologia
                commessa_obj.cronologia += 1
                commessa_obj.msg_id = str(uuid.uuid4())
                commessa_obj.ruolo = message_role.TOOL
                commessa_obj.testo = predict.model_dump_json()
                commessa_obj.tool_name = function_name
                commessa_obj.status = "read"

                ris_tool = self.save2DB(commessa_obj)

                
                if ris_tool == False:
                        return False
                
                
                # 4.e salvare messaggio risposta chatbot nel db
                commessa_obj.ruolo = message_role.ASSISTANT
                commessa_obj.tool_name = None
                commessa_obj.cel_mittente = self.config.whatsapp.phone_id
                commessa_obj.cel_destinatario = msg_recived.sender
                commessa_obj.msg_id = msg_id
                commessa_obj.testo = message_2_send
                commessa_obj.last_modify = datetime.now(tz=TIMEZONE)
                commessa_obj.cronologia += 1

                ris_bot = self.save2DB(commessa_obj)

                if ris_bot == False:
                        return False
                
            elif function_name == "push_data":

                # controlliamo la variabile
                ris_push = chat_response['tool_calls']['arguments']['push']
                if ris_push == False:

                    self.logger.debug(f"il chat model non ha permesso il salvataggio della commessa n {commessa_obj.commessa}")
                    return False
                # 1. salviamo il messaggio del utente
                commessa_obj.msg_id = msg_recived.msg_id
                commessa_obj.ruolo = message_role.USER
                commessa_obj.cel_mittente = msg_recived.sender
                commessa_obj.cel_destinatario = self.config.whatsapp.phone_id
                commessa_obj.testo = body
                commessa_obj.last_modify = msg_recived.send_time
                commessa_obj.ore_ordinarie = commessa_df['ore_ordinarie']
                commessa_obj.ore_straordinarie = commessa_df['ore_straordinarie']
                commessa_obj.ore_viaggio = commessa_df['ore_viaggio']
                commessa_obj.find_inefficienza = commessa_df['find_inefficienza']
                commessa_obj.inefficency = commessa_df['inefficency']
                commessa_obj.inefficency_type = commessa_df['inefficency_type']
                commessa_obj.inefficency_note = commessa_df['inefficency_note']
                commessa_obj.note_commessa = commessa_df['note_commessa']
                commessa_obj.status = "read"

                ris_utente = self.save2DB(commessa_obj)
                
                if ris_utente == False:
                    return False
                # 2. salviamo anche la risposta del tool
                commessa_obj.cronologia += 1
                commessa_obj.msg_id = str(uuid.uuid4())
                commessa_obj.ruolo = message_role.TOOL
                commessa_obj.testo = str(chat_response)
                commessa_obj.tool_name = function_name
                commessa_obj.status = "read"

                ris_tool = self.save2DB(commessa_obj)

                
                if ris_tool == False:
                        return False

                # 3. facciamo l'update nel db della commessa
                ris_update = self.db.updateData(commessa_obj)

                if ris_update == False:
                    self.logger.debug(f"errore nel salvataggio nel db della commessa {commessa_obj.commessa}")
                    return False
                # 2. facciamo l'update del codice intervento con complete
                ris = self.message_db.update(table_name=self.config.database.postgres_msg_table, where_field="codice_intervento", where_value=commessa_obj.codice_intervento, set_field="complete", set_value=True)
                if ris == False:
                    return False
                # inviamo un messagio di conferma all'utente:
                msg_2_send = f"La commessa *{commessa_obj.commessa}* è stata aggiornata correttamente nel database!"
                response = self.send_message(to=msg_recived.sender, message=msg_2_send)
                if response == None:
                    return False

                
        else:
            # non ha chiamato nessun tool
            self.logger.error(f"il chat model per la commessa {commessa_obj.commessa} non ha usato nessun tool ma dovrebbe utilizzarlo")
            return False
            
        
        
        return True



    def BuildContext(self, data:pd.DataFrame)-> list[dict]|None:
        """
        Funzione che costruire il contesto dato le righe della tabella per una specifica commessa.
        -df(DataFrame): un df che contiene i messaggi precedentemente avuti per la commessa specifica

        l'output sarà simile a questo:
        [{
            'role': 'user',
            'content': 'ciao ieri ho svolto 6 ore di lavoro'
        },
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'type': 'function', 'function': {'name': 'extractor_expert', 'arguments': {'summary': 'ha lavorato 6 ore'}}}]
        },
        {
            'role': 'tool',
            'tool_name': 'extractor_expert'
            'content': '{"ore_ordinarie":6.0,"ore_straordinarie":0.0,"ore_viaggio":0.0,"innefficienza":false,"note":"","commessa":"","risposta_singola":""}'
        },
        {
            'role': 'assistant',
            'content': 'Ho riconosciuto i seguenti campi: ore effettuate:6.0, ecc...'
        }]
        """
        # copiamo i dati
        df = data.copy()
        output = []

        # rimuoviamo il primo elemento che è il template e non serve
        df.drop([0], inplace=True)
        
        # se abbiamo solo il messaggio template abbiamo già finito possiamo passare la list vuota 
        if len(df) == 0:
            return output
        
        # per ogni riga
        for _,row in df.iterrows():
            # per ogni riga ci serve il ruolo il testo(content) e il tool_name
            match row.ruolo:
                case 'tool':
                    # caso tool 
                    output.append({
                        "role": "tool",
                        "tool_name": row.tool_name,
                        "content": row.testo
                    })
                case _:
                    # caso default ( user, system or assistant)
                    # facciamo un check che sia uno degli enum 
                    if row.ruolo not in message_role:
                        self.logger.error(f"per il tecnico con matricola {data.iloc[0].matricola_tecnico} c'è un ruolo non ammissibile: {row.ruolo} ")
                        return None
                    # scriviamo
                    output.append({
                        "role": row.ruolo,
                        "content": row.testo
                    })
        
        return output
    
    def checkLevel(self,msg_recived:Event)->pd.DataFrame|None:
        """
        Funzione che indentifica a che livello siamo della conversazione. 

        
        
        :param msg_recived: il messaggio ricevuto
        :type msg_recived: Event
        :return: l'ultima riga della commessa trovata per il tecnico specifico, oppure un numero di righe pari a n se il tecnico ha n commesse attive per il singolo giorno
        :rtype: DataFrame | None
        """
        results = self.message_db.find(table_name=self.config.database.postgres_msg_table, by="cel_mittente", value=msg_recived.sender)
        # escludiamo le commesse già completate 
        results = results[results['complete'] != True] 
        if results is None:
            self.error_message(to=msg_recived.sender)
            return None
        self.logger.debug(f"Risultati trovati nel db per il mittente {msg_recived.sender}: {results.shape}")

        # contatore per aumentare di uno la cronologia
        plus = 0 
        if len(results) == 0:

            self.logger.debug("nessun occorrenza trovata vuol dire che non ha scritto ancora nulla -> ricerchiamo per destinatario per trovare la commessa")
            results = self.message_db.find(table_name=self.config.database.postgres_msg_table, by="cel_destinatario", value=msg_recived.sender)
            
            # escludiamo le commesse già completate 
            results = results[results['complete'] != True] 
            if results is None:
                self.error_message(to=msg_recived.sender)
                return None
            #self.logger.debug(f"Risultati trovati nel db per il destinatario {msg_recived.sender}: {results.shape}")
            if len(results) == 0:
                # non ci dobbiamo interessare a questi messaggi da numeri che non compaiono nel db
                self.logger.info(f"arrivato un messaggio da {msg_recived.sender} che non compare nel database dei Messaggi. Il messaggio verrà ignorato")
                return 
        else:
            self.logger.debug("trovato più di un messaggio inviato dall'utente. Secondo lo schema sarà a +2 la cronologi massima (perchè l'assistente risponde sempre)")
            plus += 1
            

        # troviamo i campi da copiare (commessa, codice intervento e data intervento) e la cronologia più alte
        max_indici = results.groupby(['codice_intervento'])['cronologia'].idxmax() 
        max_commesse = results.loc[max_indici]
        
        
        # controlliamo se la cronologia è a zero 
        res = max_commesse.loc[max_commesse['cronologia'] == 0]

        #self.logger.debug(f"max_commesse len: {len(max_commesse)}, res len: {len(res)}")
        
        # a seconda del caso in cui abbiamo trovato un unica commessa per l'operatore oppure più commesse
        commessa_df = res if len(res) == 1 else max_commesse
        
        # incrementiamo già per il nuovo messaggio arrivato
        commessa_df.cronologia += 1 + plus

        return commessa_df
        

    def manageRecivedMessage(self, msg_recived:Event):
        """
        Funzione per gestire i messaggi ricevuti dagli utenti.
        - msg_recived(str): il messaggio arrivato dall'utente.
        """
        self.logger.info(f"messagio ricevuto del tipo:{msg_recived.msg_type} dal numero: {msg_recived.sender}")

        # gestione dei messaggi
        if msg_recived.msg_type == 'text':
            # messaggio di testo
            body = msg_recived.msg_body
        elif msg_recived.msg_type == 'audio':
            body = self.handleAudioMessage(msg_recived)
            if body == None:
                # avvenuto un errore, usciamo
                return
        else:
            # nessun altr tipo che ci interessa
            message = "Non sono in grado di processare immagini o altri file. Per favore invia un messaggio o un audio"
            response = self.send_message(to=msg_recived.sender, message=message)
            return 
        
        self.logger.debug(f"messaggio ricevuto: {body}")

        # controlliamo a che livello siamo:
        commessa_df = self.checkLevel(msg_recived)
        if commessa_df is None:
            return
        
        self.logger.debug(f"commessa/e: {commessa_df.commessa.iloc[0]}\tcronologia/e: {commessa_df.cronologia.iloc[0]}")

        # il chat modello riceve il body del messaggio + tutta la sequenza di cronologia
        # 1) se abbimo solo una commessa per un tecnico -> non è un problema in teoria perché si fa un select dei messaggi e le si converte nel formato: assistant... user... tool... 
        # 2) se abbimo due o più commesse per un tecnico -> si può fare un primo controllo senza contesto -> se si individua la commessa -> ritorniamo al caso 1 -> se no diciamo che non riusciamo a capire a che commessa si fa riferimento
        #   2.1) in questo caso per farlo funziona almeno correttamente servirebbe scrivere al tecnico: "se hai due commesse devi inviare due messaggi separati e includere il codice della commessa nel messaggio"
        
            
        if len(commessa_df) == 1:
            # caso con commessa univoca
            # convertiamo il dataframe in list of dicts per comodita
            commessa_obj = commessa_df.to_dict("records")[0]
            self.logger.debug(f"abbiamo un unica commessa, con i dati : {commessa_obj}")
            ris = self.analyzeMessage(commessa_obj, body, msg_recived)
            if ris == False:
                self.error_message(to=msg_recived.sender)
        else:
            # abbiamo più di una commessa per lo stesso tecnico
            self.logger.info(f"abbiamo più di una commessa per lo stesso tecnico con matricola: {commessa_df.matricola_tecnico.iloc[0]}")
            commesse_find = []
            for idx in range(len(commessa_df)):
                row = commessa_df.iloc[idx]
                if row.commessa in body:
                    # aggiungiamo quando troviamo una corrispondenza nel body
                    commesse_find.append(idx)
            
            match len(commesse_find):
                case 1:
                    # passiamo per analizzare 
                    commessa_obj = commessa_df.iloc[commesse_find[0]].to_dict()
                    ris = self.analyzeMessage(commessa_obj, body, msg_recived)
                    if ris == False:
                        self.error_message(to=msg_recived.sender)
                case 0:
                    # se non troviamo nessuna commessa scriviamo all'utente di scrivercela
                    msg_2_send = f"Non siamo stati ingrado di indetificare la commessa. A sistema sono state trovate queste commesse svolte:\n*{list(commessa_df.commessa)}*\nRiscrivere il messaggio includendo la commessa specifica"
                    response = self.send_message(to=msg_recived.sender, message=msg_2_send)

                    if response == None:
                        self.error_message(to=msg_recived.sender)
                case _:
                    # abbiamo trovato più di una commessa nel messaggio. chiediamo di separarle
                    msg_2_send = f"Abbiamo trovato più di una commessa nel messaggio. Separare in due messaggi diversi le commesse"
                    response = self.send_message(to=msg_recived.sender, message=msg_2_send)
                    if response == None:
                        self.error_message(to=msg_recived.sender)
                    
            

       
    def sendMessageAgain(self, commessa_obj:MessaggioTecnici, evento:Event, fail_count:int):
        """
        Metodo per inviare nuovamente un messaggio che ha avuto errore
        
        :param commessa_obj: commessa da inviare nuovamente
        :type commessa_obj: Messaggio
        :param evento: evento del messaggio
        :type evento: Event
        :param fail_count: counter dei fail
        :type fail_count: int
        """
        # controlla la cronologia per vedere se è un primo messaggio
        if commessa_obj.cronologia == 0: 
            # se è il primo ne creiamo un altro ed eliminiamo la seguente riga
            # ri prendiamo il nome del cliente
                 
            try:
                data = self.db.find_intervento(commessa_obj.codice_intervento)
                if data.empty:
                    self.logger.info(f"il tecnico {commessa_obj.matricola_tecnico} non ha più questo intervento da realizzare, vuol dire che l'ha già inserita. Possiamo anche non inviare il messaggio") 
                    # cancelliamo la riga dal db
                    ris = self.message_db.delete(table_name=self.config.database.postgres_msg_table, where_field="msg_id", where_value=evento.msg_id)
                    return
                
                row = {}
                row['TECNICO_MOBILE'] = commessa_obj.cel_destinatario
                row['COMMESSA'] = commessa_obj.commessa
                row['DATA_INTERVENTO'] = data.iloc[0]["DATA_INTERVENTO"]
                row['DES_CLIENTE'] = data.iloc[0]["DES_CLIENTE"]
                row['MATR'] = commessa_obj.matricola_tecnico
                row['CODICE_INTERVENTO'] = commessa_obj.codice_intervento
            except Exception as e:
                self.logger.error(f"errore nella creazione della riga originale per inviare il messaggio: {str(e)}")
            self.logger.warning(f"superato la creazione della riga orginale con valori: \n{row}\n")

            # cancelliamo prima la riga dal db
            ris = self.message_db.delete(table_name=self.config.database.postgres_msg_table, where_field="msg_id", where_value=evento.msg_id)
            if ris == False:
                return
            template = self._create_template(row)
            self.logger.warning(f"Il numero {commessa_obj.cel_destinatario} ha il template da passare: \n{template}\n")
            if 'errore' in template:
                return 
            self.send_message(to=commessa_obj.cel_destinatario, template=template, fail_counter=fail_count)
        else:
            # se non è il primo copiamo il testo che era dentro a questa commessa e tutti gli altri dati e lo inviamo nuovamente cancellando la seguente riga
            response = self.send_message(to=commessa_obj.cel_destinatario, message=commessa_obj.testo)
            if response == None:
                return 
            id_msg = response['messages'][0]['id']
            commessa_obj.status = ""
            commessa_obj.msg_id = id_msg
            commessa_obj.last_modify = datetime.now(tz=TIMEZONE)
            commessa_obj.failed_count = fail_count
            ris_bot = self.save2DB(commessa_obj)
            if ris_bot == False:
                    return 
            # cancelliamo la vecchia riga
            ris = self.message_db.delete(table_name=self.config.database.postgres_msg_table, where_field="msg_id", where_value=evento.msg_id)
            if ris == False:
                return
    
    def manageMessageAsync(self, data):
        """
        Metodo che processa il messaggio successivamente per evitare di ricevere ulteriori webhook da Whatsapp
        """
        # wrap in Event
        try:
            evento = Event(data)
        except Exception as err:
            self.logger.error(f"Errore nella lettura della request:{data}\n\n Errore:{str(err)}")
            return
        if evento.event_type == 'event':
            self.logger.info(f"Evento ricevuto da {evento.sender}. per il messagio: {evento.msg_id}. il messagio è: {evento.msg_status}")
            if evento.msg_status == 'failed':
                
                self.logger.warning(f"il messaggio al numero {evento.sender} ha avuto un errore, si proverà a rinviare il messaggio")
                # troviamo la riga nel db
                result = self.message_db.find(table_name=self.config.database.postgres_msg_table, by="msg_id", value=evento.msg_id)
                if result is None:
                    self.logger.error(f"Non trovato nessun messaggio con id {evento.msg_id} per il numero {evento.sender}")
                    return 

                # convertiamo il dataframe in list of dicts
                commessa_df = result.to_dict("records")[0]
                self.logger.warning(f"trovata l'intervento da inviare nuovamente: \n {commessa_df}\n")
                # convertiamo il dict in Messaggio così da lavorarci meglio
                # prima rimuoviamo tutti i campi che sono null
                commessa_df_clean = {k:v for k,v in commessa_df.items() if v != None}
                
                # aggiungere timezone aware per last_modify
                try:
                    commessa_df_clean['last_modify'] = TIMEZONE.localize(commessa_df_clean['last_modify'])
                    commessa_obj = MessaggioTecnici(**commessa_df_clean)
                except Exception as err:
                    self.logger.error(f"Errore nella validazione del messaggio da salvare nel db inviato per tecnico {commessa_df['matricola_tecnico']} numero di cronologia: {str(err)}")
                    return 

                self.logger.warning(f"convertito il messaggio l'elemento che bisogna inviare nuovamente: \n{commessa_obj}\n")
                # controlliamo quante volte il messaggio fa fail
                match commessa_obj.failed_count:
                    case 0:
                        # prima volta che capita
                        self.logger.debug(f"per il numero {evento.sender} si prova nuovamente a inviare il primo messaggio")
                        self.sendMessageAgain(commessa_obj, evento, fail_count=1)
                        
                    case 1:
                        # avvenuto già una volta aspettiamo cinque minuti prima di continuare
                        self.logger.info(f"per il numero {commessa_obj.cel_destinatario} l invio di un messaggio non è andato a buon fine, aspettare cinque minuti e inviare nuovamente")
                        time.sleep(60*5)
                        self.sendMessageAgain(commessa_obj, evento, fail_count=2)
                    case _:
                        self.logger.warning(f"per il numero {commessa_obj.cel_destinatario} un messaggio è stato di nuovo negato. Adesso si smette di inviare il messaggio")
                        return 

            else:
                # facciamo l'update dello status
                self.message_db.update(table_name=self.config.database.postgres_msg_table, where_field="msg_id", where_value=evento.msg_id, set_field="status", set_value=evento.msg_status)
        elif evento.event_type == 'message':
            self.manageRecivedMessage(evento)


    def _routes(self):
        """
        Catch di tutte le routes 
        """

        @self.app.route('/stop', methods=['POST'])
        def stop_app():
            """
            per stoppare l'app Flask utilizzando un authentication token, con una request:
            curl.exe -X POST http://127.0.0.1:5000/stop -H "Authorization: Bearer <token>"
            """
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'status': 'error', 'message': 'Token mancante'}), 401
            
            # estraiamo il token
            token_type, token = auth_header.split(" ")
            if token_type.lower() != "bearer":
                return jsonify({'status': 'error', 'message': 'formato di auth non valido. Usare Bearer <token>'}), 401
            
            # verifichiamo che sia lo stesso del verify token
            if token != self.config.whatsapp.verify_token:
                return jsonify({'status': 'error', 'message': 'Token non valido'}), 403

            
            self.stop()
            return jsonify({"status" : "stop", "message" : "Server is stopped"})
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status" : "ok", "port" : self.config.whatsapp.port})
        
        @self.app.route('/monitor', methods=['GET'])
        def monitor():
            """Endpoint per salvare tutti i messaggi ricevuti fino ad ora nel database"""
            # salviamo il log nel db
            try:
                ris = self.message_db.export_to_sql_server(self.config.database.postgres_msg_table)
                if ris == False:
                    return jsonify({"status": "error", "message": "errore nel salvataggio dei messaggi nel db"}), 500
            except Exception as e:
                return jsonify({"status": "error", "message": "errore nel salvataggio dei messaggi nel db"}), 500
            
            return jsonify({"status" : "ok", "message" : "messaggi salvati correttamente nel database"})

            
        
        @self.app.route('/webhook/tecnico', methods=['GET','POST'])
        def webhook():  
            """
            Webhook per ricevere i messaggi e ricevere lo status di un messaggio che si ha inviato.
            """
            if request.method == 'GET':
                # verificare webhook (setup)
                mode = request.args.get('hub.mode')
                token = request.args.get('hub.verify_token')
                challenge = request.args.get('hub.challenge')

                if mode == 'subscribe' and token == self.config.whatsapp.verify_token:
                    self.logger.info("Webhook verificata con successo")
                    return challenge, 200
                else: 
                    return 'Forbidden', 403            
            else:
                # ricezione messaggio
                data = request.json
                # avvio il thread in background 
                future = self.webhook_executor.submit(self.manageMessageAsync, data)
                # rispondi immediatamente 
                return jsonify({'status': 'success'}), 200
                
                
            
        
        





    