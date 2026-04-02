from schema.db import DB_Messaggi
from omegaconf import OmegaConf
from utils.logger import Logger
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore 
from apscheduler.executors.pool import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor as ConcurrentThreadPoolExecutor
from sqlalchemy import create_engine, inspect, text
import json
from datetime import datetime, timedelta
import pickle
import json
from datetime import datetime
from flask import Flask, jsonify, request
import numpy as np
import time

logger = Logger(save=False, consoleLevel="DEBUG").getLogger()
conf = OmegaConf.load("config_follow_up.yaml")

data=[{'cel':1, 'msg':'ciao','order':0, 'data_intervento': int(time.time()) + 900},{'cel':1, 'msg':'ciao 1','order':1, 'data_intervento': int(time.time()) + 600},{'cel':1, 'msg':'ciao 2','order':2, 'data_intervento': int(time.time()) + 300},{'cel':1, 'msg':'ciao 3','order':3, 'data_intervento': int(time.time())},{'cel':2, 'msg':'ciao da cel due','order':0, 'data_intervento': int(time.time())}]

def task_di_esempio():
    print(f"Invio messaggio")


app = Flask(__name__)


def init_scheduler(app):
        job_stores = {'default': SQLAlchemyJobStore(url=conf.database.postgres_jobstore)}
        scheduler = BackgroundScheduler(jobstores=job_stores)
        scheduler.start()
        return scheduler

def add_jobs(scheduler):
    numeri_commerciali = {}
    for row in data:
        numero = row['cel']
        if numero in numeri_commerciali:
            numeri_commerciali[numero].append((row['order'], row['data_intervento']))
        else:
            numeri_commerciali[numero] = [(row['order'], row['data_intervento'])]

    
    jobs = scheduler.get_jobs()
    for num,occ in numeri_commerciali.items():
            # inviamo uno ogni 5 minuti (esempio) a seconda se troviamo degli elementi con lo stesso cod offerta e numero
            
            for idx,obj in enumerate(occ):
                cod_offerta, data_unix = obj
                FindElement = False
                for job in jobs:
                    parts = job.id.split('_')
                    print(parts)
                    if parts[1] == num and parts[2] == cod_offerta:
                        # elemento trovato, non dobbiamo metterlo
                        FindElement = True
                        break
                
                if not FindElement:
                    delta = idx*5
                    print(f'delta time : {delta}')
                    when = datetime.now() + timedelta(minutes=delta)
                    id_use = f'process_{num}_{cod_offerta}_{data_unix}'
                    print(f'id da utilizzare : {id_use}')
                    scheduler.add_job(
                        task_di_esempio,
                        'date',
                        run_date=when,
                        id=id_use,
                        name='Task prova',
                        replace_existing=True
                        )

        
def schedula_job_per_codice(scheduler, numero,interventi, task_func):
    
    jobs = scheduler.get_jobs()
    existing_ids = [job.id for job in jobs]
    num_str = str(numero)

    job_count_for_this_number = sum(
        1 for job_id in existing_ids 
        if f'process_{num_str}_' in job_id
    )

    timestamp_for_this_number = []
    for job_id in existing_ids:
        if f'process_{num_str}_' in job_id:
            parts = job_id.split('_')
            timestamp_for_this_number.append(int(parts[3]))


    jobs_added = 0
    for idx, (cod_offerta,data_unix) in enumerate(interventi):

        print(f'cod_offerta: {cod_offerta} e data_unix: {data_unix}')
        id_use = f'process_{num_str}_{cod_offerta}_{data_unix}'

        if id_use in existing_ids:
            continue
            
        # controlliamo se quello che abbiamo è il più recente di tutti
        is_most_recent = True
        for timestamp in timestamp_for_this_number:
            if timestamp > data_unix:
                is_most_recent = False
                break
        
        if not is_most_recent and timestamp_for_this_number:
            continue

        total_job_index = job_count_for_this_number + jobs_added 
        delta_minuti = total_job_index * 5
        when = datetime.now() + timedelta(minutes=delta_minuti)
        
        try:
            scheduler.add_job(
                task_func,
                'date',
                run_date=when,
                id=id_use,
                name=f'Task {num_str} - {cod_offerta}',
                replace_existing=True
            )
        except Exception as e:
            print(f"errore: {e}\n")
        jobs_added += 1
    
        



scheduler = init_scheduler(app)
#add_jobs(scheduler)


@app.route('/jobs')
def list_jobs():
    jobs = scheduler.get_jobs()
    return {job.id: str(job) for job in jobs}

@app.route('/add', methods=['POST'])
def add_jobing():
    data = request.get_json()

    numeri_commerciali = {}
    for row in data:
        numero = row['cel']
        if numero in numeri_commerciali:
            numeri_commerciali[numero].append((row['order'], row['data_intervento']))
        else:
            numeri_commerciali[numero] = [(row['order'], row['data_intervento'])]

    print(numeri_commerciali)
    

    for num,occ in numeri_commerciali.items():
            # inviamo uno ogni 5 minuti (esempio) a seconda se troviamo degli elementi con lo stesso cod offerta e numer
            schedula_job_per_codice(scheduler,num,occ,task_di_esempio)

    return jsonify({"status" : "ok"})


@app.route('/stop')
def shutdown_scheduler():
    scheduler.shutdown()


if __name__ == "__main__":
    app.run(port=6001)
    

    """ with engine.connect() as conn:
        result = conn.execute(text("SELECT id, job_state, next_run_time   FROM apscheduler_jobs"))
        for job_id, job_state_binary, next_run in result:
            print(f"\n📍 Job ID: {job_id}")
            next_run_dt = datetime.fromtimestamp(next_run)
            print(f"   Next run: {next_run_dt}")

            if job_state_binary:
                try:
                    # Deserializza il job_state
                    job_obj = pickle.loads(job_state_binary)

                    print(f"   Job State: {job_obj}")
                    print(f"      - Func: {job_obj.func_ref}")
                    print(f"      - Trigger: {job_obj.trigger}")
                    print(f"      - Args: {job_obj.args}")
                    print(f"      - Kwargs: {job_obj.kwargs}")

                except Exception as e:
                    print(f"   ❌ Errore deserializzazione: {e}") """







