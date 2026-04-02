import pandas as pd
from omegaconf import DictConfig
from utils.logger import Logger
from schema.postgres_models import SQLModel, MessaggioTecnici, Etichetta, MessaggioFollowUp
import psycopg as psy
import pyodbc
from datetime import datetime,timedelta
from pytz import timezone
from abc import abstractmethod
import os

CSV_FILENAME_TECNICI='./data/data_prova_tecnici.csv'
CSV_FILENAME_FOLLOW='./data/data_prova_follow_up.csv'

COLUMNS=[
    'CODICE_INTERVENTO','COMMESSA','DES_CLIENTE','TECNICO','TECNICO_TEL','TECNICO_MOBILE','DATA_INTERVENTO',
    'ORE_ORDINARIE','ORE_STRAORDINARIE','ORE_VIAGGIO','INEFFICIENCY','INEFFICIENCY_TYPE','HOURS_BY_CLIENT','MATR','NOTE'
]

COLUMNS_FOLLOW_UP=['IdFollowUp','Cliente','IdOfferta','CodOfferta','titolo','DataUltimoContatto','Nome','Cognome','Telefono','Cellulare','Link', 'rn']

# per ora che le righe non vanno via da lyra, utilizzo questo meccanisco per aggiornare alla successiva offerta
COD_OFFERTA_EXCLUDED = ['260033']
logger = Logger(save=True).getLogger()



class DB_Followup:
    """
    Classe per fare il fetch dei follow up  
    """
    def find_offerta(self, offerta:str):
        """Funzione che trova la riga tramite la Codofferta del giorno interessato"""
        data = None
        try:
                conn = pyodbc.connect(self.config.database.sqlserver_conn)
                cursor = conn.cursor()
                # selezionare il giorno precedente 
                selected_date = datetime.now(tz=timezone('Europe/Rome'))
                # query the specific day if there isn't data
                query = """SELECT TOP (1000) [IdFollowUp]
                            ,[Cliente]
                            ,[IdOfferta]
                            ,[CodOfferta]
                            ,[titolo]
                            ,[DataUltimoContatto]
                            ,[Nome]
                            ,[Cognome]
                            ,[Telefono]
                            ,[Cellulare]
                            ,[Link]
                            FROM [AI_REPO].[dbo].[vSITI_FollowUp_Ultimi_60Giorni] WHERE DATEDIFF(day,[DataUltimoContatto],'{}') > 60 AND [Cellulare] IS NOT NULL AND [CodOfferta] = '{}'
                """.format(selected_date.strftime("%Y-%m-%d"), offerta)
                results =cursor.execute(query).fetchall()
                logger.debug(f"risultati della query fetch per il giorno {selected_date.strftime("%Y-%m-%d")} e per l'offerta {offerta}: {len(results)} \n{results}")
                data = None
                data = pd.DataFrame([tuple(row) for row in results], columns=COLUMNS_FOLLOW_UP)
                logger.debug(f"data: {data.head()}\n{data.shape}\n{data.columns}")
                    

        except Exception as e:
            logger.error(f"Errore nel fare il check dei dati: {e}")
            return False
        
        return data

   
    
    def _upload_data(self):
        """
        Funzione per fare il fetch
        """
        data = None
        if self.config.database.is_db:
            # use db 
            try:
                conn = pyodbc.connect(self.config.database.sqlserver_conn)
                cursor = conn.cursor()
                # selezionare il giorno stesso 
                selected_date = datetime.now(tz=timezone('Europe/Rome')) 

                # aggiungiamo il filtro per cambiare l'offerta se é gia stata fatta
                exclude_filter = ""
                if COD_OFFERTA_EXCLUDED:
                    placeholders = ", ".join(f"'{c}'" for c in COD_OFFERTA_EXCLUDED)
                    exclude_filter = f"AND [CodOfferta] NOT IN ({placeholders})"
                
                # il fetch viene fatto ogi giorno e si seleziona solo l'offerta piú recente
                query =""" 
                        WITH cte AS (
                                SELECT TOP (1000)  
                                    [IdFollowUp], [Cliente], [IdOfferta], [CodOfferta], [titolo],
                                    [DataUltimoContatto], [Nome], [Cognome], [Telefono], [Cellulare], [Link],
                                    ROW_NUMBER() OVER (PARTITION BY [Cellulare] ORDER BY [DataUltimoContatto] DESC) AS rn
                                FROM [AI_REPO].[dbo].[vSITI_FollowUp_Ultimi_60Giorni] 
                                WHERE DATEDIFF(day,[DataUltimoContatto],'{}') > 60  
                                  AND [Cellulare] IS NOT NULL
                                  {}
                            )
                            SELECT * FROM cte WHERE rn = 1
                """.format(selected_date.strftime("%Y-%m-%d"), exclude_filter)
                results =cursor.execute(query.strip()).fetchall()
                logger.debug(f"risultati della query fetch per il giorno {selected_date.strftime("%Y-%m-%d")}: {len(results)} \n{results}")
                data = None
                data = pd.DataFrame([tuple(row) for row in results], columns=COLUMNS_FOLLOW_UP)
                logger.debug(f"data: {data.head()}\n{data.shape}\n{data.columns}")
                    

            except Exception as e:
                logger.error(f"Errore nel fare il check dei dati: {e}")
                return []
            
        else:
            # use csv file
            data = pd.read_csv(CSV_FILENAME_FOLLOW)
            logger.info(f'data returned {data.shape}. Columns: {data.columns}')
            # convert the date in datetime64
            #data['DATA_INTERVENTO'] = pd.to_datetime(data['DATA_INTERVENTO'], format='%Y-%m-%d')
            data['DataUltimoContatto'] = pd.to_datetime(data['DataUltimoContatto'])
            # aggiungiamo il timezone
            #data['DATA_INTERVENTO'] = data['DATA_INTERVENTO'].dt.tz_localize('Europe/Rome')

            # poi convertiamo in stringa
            #data['DATA_INTERVENTO'] = data['DATA_INTERVENTO'].dt.strftime('%d/%m/%Y')
            
            
            # convertiamo  in stringa
            data['Cellulare'] = data['Cellulare'].astype(str)
            # rimuovi il + davanti al numero e rimuovi gli spazi
            data['Cellulare'] = data['Cellulare'].apply(process_cel)

            data['CodOfferta'] = data['CodOfferta'].astype(str)
            data['titolo'] = data['titolo'].astype(str)
            
            

        if data is None:
            logger.error('dati non caricati')
            return False

        logger.info(f'shape: {data.shape}')
        return data


    
    
    def updateData(self, row:MessaggioFollowUp)->bool:
        """
        Funzione che fa l'update della riga della commessa nel db finale 
        """
                
        
        if self.config.database.is_db:
            
            
            TMP_FILE = "history/save_follow_up/followups.csv"
            import os
            #FIXME: con la procedura nuova
            try:
                referente_parts = row.referente.split(' ', 1)
                data_2_save = {
                    'CodOfferta': [row.cod_offerta],
                    'NOME': [referente_parts[0]],
                    'COGNOME': [referente_parts[1] if len(referente_parts) > 1 else ''],
                    'LINK': [row.link]
                }
                if row.prob_acquisizione and row.prob_acquisizione != 0:
                    data_2_save['prob_acquisizione'] = [row.prob_acquisizione]
                if row.data_consegna and row.data_consegna != '':
                    data_2_save['data_consegna'] = [row.data_consegna]
                if row.prezzo_vendita and row.prezzo_vendita != 0:
                    data_2_save['prezzo_vendita'] = [row.prezzo_vendita]
                if row.note and row.note != '':
                    data_2_save['note'] = [row.note]

                os.makedirs(os.path.dirname(TMP_FILE), exist_ok=True)

                if os.path.exists(TMP_FILE):
                    data = pd.read_csv(TMP_FILE)
                    data = pd.concat([data, pd.DataFrame.from_dict(data_2_save)], ignore_index=True)
                else:
                    data = pd.DataFrame.from_dict(data_2_save)

                data.to_csv(TMP_FILE, sep=",", index=False)

            except Exception as e:
                logger.error(f"Errore nella salvare l'offerta {row.cod_offerta}: {e}")
                return False
        else:
            # use csv 
            logger.debug(f"offerta da salvare: {row}")
            return True
            self.data.loc[self.data['CodOfferta'].astype(str) == row.cod_offerta, 'prob_acquisizione'] = row.prob_acquisizione 
            self.data.loc[self.data['CodOfferta'].astype(str) == row.cod_offerta, 'data_consegna'] = row.data_consegna 
            self.data.loc[self.data['CodOfferta'].astype(str) == row.cod_offerta, 'prezzo_vendita'] = row.prezzo_vendita
            self.data.loc[self.data['CodOfferta'].astype(str) == row.cod_offerta, 'note'] = row.note


            logger.debug(f"dati aggiornati: {self.data[['prob_acquisizione','data_consegna', 'prezzo_vendita', 'note']]}")

            # salviamo nel file
            self.data.to_csv(CSV_FILENAME_FOLLOW)
        return True 

    def __init__(self, config: DictConfig):
        self.config = config
        self.data = self._upload_data()

    def  is_empty(self):
        return len(self.data) == 0 


class DB_Commesse:
    """
    Classe per fare il fetch delle commesse
    """

    def find_intervento(self, intervento:str):
        """Funzione che trova la riga tramite l'intervento del giorno interessato"""
        data = None
        try:
                conn = pyodbc.connect(self.config.database.sqlserver_conn)
                cursor = conn.cursor()
                # selezionare il giorno precedente 
                selected_date = datetime.now(tz=timezone('Europe/Rome')) + timedelta(days=-1)
                # query the specific day if there isn't data
                query = """SELECT [CODICE_INTERVENTO]
                            ,[COMMESSA]
                            ,[DES_CLIENTE]
                            ,[TECNICO]
                            ,[TECNICO_TEL]
                            ,[TECNICO_MOBILE]
                            ,[DATA_INTERVENTO]
                            ,[ORE_ORDINARIE]
                            ,[ORE_STRAORDINARIE]
                            ,[ORE_VIAGGIO]
                            ,[INEFFICIENCY]
                            ,[INEFFICIENCY_TYPE]
                            ,[HOURS_BY_CLIENT]
                            ,[MATR]
                            ,[NOTE]  
                           FROM [AI_REPO].[dbo].[vSITI_REPORT_GIORNI_TECNICI] WHERE [DATA_INTERVENTO] = '{}' AND 
                                [ORE_VIAGGIO]  IS NULL AND [ORE_STRAORDINARIE] IS NULL AND [ORE_ORDINARIE] IS NULL
                                AND [NOTE] IS NULL AND [INEFFICIENCY] IS NULL AND LEFT([MATR],1) < 'B' AND [MATR] <> '' AND [TECNICO_MOBILE] IS NOT NULL AND [CODICE_INTERVENTO] = '{}'
                """.format(selected_date.strftime("%Y-%m-%d"), intervento)
                results =cursor.execute(query).fetchall()
                logger.debug(f"risultati della query fetch per il giorno {selected_date.strftime("%Y-%m-%d")} e per l'intervento {intervento}: {len(results)} \n{results}")
                data = None
                data = pd.DataFrame([tuple(row) for row in results], columns=COLUMNS)
                logger.debug(f"data: {data.head()}\n{data.shape}\n{data.columns}")
                    

        except Exception as e:
            logger.error(f"Errore nel fare il check dei dati: {e}")
            return False
        
        return data

   
    
    def _upload_data(self):
        """
        Funzione per fare il fetch
        """
        data = None
        if self.config.database.is_db:
            # use db 
            try:
                conn = pyodbc.connect(self.config.database.sqlserver_conn)
                cursor = conn.cursor()
                # selezionare il giorno precedente 
                selected_date = datetime.now(tz=timezone('Europe/Rome')) + timedelta(days=-1)
                # query the specific day if there isn't data

                
                query = """SELECT [CODICE_INTERVENTO]
                            ,[COMMESSA]
                            ,[DES_CLIENTE]
                            ,[TECNICO]
                            ,[TECNICO_TEL]
                            ,[TECNICO_MOBILE]
                            ,[DATA_INTERVENTO]
                            ,[ORE_ORDINARIE]
                            ,[ORE_STRAORDINARIE]
                            ,[ORE_VIAGGIO]
                            ,[INEFFICIENCY]
                            ,[INEFFICIENCY_TYPE]
                            ,[HOURS_BY_CLIENT]
                            ,[MATR]
                            ,[NOTE]  
                           FROM [AI_REPO].[dbo].[vSITI_REPORT_GIORNI_TECNICI] WHERE [DATA_INTERVENTO] = '{}' AND 
                                [ORE_VIAGGIO]  IS NULL AND [ORE_STRAORDINARIE] IS NULL AND [ORE_ORDINARIE] IS NULL
                                AND [NOTE] IS NULL AND [INEFFICIENCY] IS NULL AND LEFT([MATR],1) < 'B' AND [MATR] <> '' AND [TECNICO_MOBILE] IS NOT NULL
                """.format(selected_date.strftime("%Y-%m-%d"))
                results =cursor.execute(query).fetchall()
                logger.debug(f"risultati della query fetch per il giorno {selected_date.strftime("%Y-%m-%d")}: {len(results)} \n{results}")
                data = None
                data = pd.DataFrame([tuple(row) for row in results], columns=COLUMNS)
                logger.debug(f"data: {data.head()}\n{data.shape}\n{data.columns}")
                    

            except Exception as e:
                logger.error(f"Errore nel fare il check dei dati: {e}")
                return False
            
        else:
            # use csv file
            data = pd.read_csv(CSV_FILENAME_TECNICI)
            logger.info(f'data returned {data.shape}. Columns: {data.columns}')
            # convert the date in datetime64
            #data['DATA_INTERVENTO'] = pd.to_datetime(data['DATA_INTERVENTO'], format='%Y-%m-%d')
            data['DATA_INTERVENTO'] = pd.to_datetime(data['DATA_INTERVENTO'])
            # aggiungiamo il timezone
            #data['DATA_INTERVENTO'] = data['DATA_INTERVENTO'].dt.tz_localize('Europe/Rome')

            # poi convertiamo in stringa
            #data['DATA_INTERVENTO'] = data['DATA_INTERVENTO'].dt.strftime('%d/%m/%Y')
            
            
            # convertiamo  in stringa
            data['TECNICO_MOBILE'] = data['TECNICO_MOBILE'].astype(str)
            # rimuovi il + davanti al numero e rimuovi gli spazi
            data['TECNICO_MOBILE'] = data['TECNICO_MOBILE'].apply(process_cel)

            data['COMMESSA'] = data['COMMESSA'].astype(str)
            data['CODICE_INTERVENTO'] = data['CODICE_INTERVENTO'].astype(str)
            data['MATR'] = data['MATR'].astype(str)
            

        if data is None:
            logger.error('dati non caricati')
            return False

        logger.info(f'shape: {data.shape}')
        return data


    
    
    def updateData(self, row:MessaggioTecnici)->bool:
        """
        Funzione che fa l'update della riga della commessa nel db finale 
        """
        # controlliamo che tutti i dati che ci servono sono presenti
        if row.commessa ==  None or row.commessa == "" and row.ore_ordinarie == row.ore_straordinarie == row.ore_viaggio == None and row.innefficienza == None and row.note_commessa == None:
            return False
        if self.config.database.is_db:
            # chiamiamo la procedura con EXEC SP_UPD0001_Aggiornamento_righe_intervento  1, 0 dove 1 è una variabile che intende la fase (Development,ecc...) e poi 0 in questo caso sarebbe il codice intervento
            try:
                conn = pyodbc.connect(self.config.database.sqlserver_conn, autocommit=True)
                cursor = conn.cursor()
                
                # settiamo a 0 se None le ore di inefficienza 
                row.inefficency = 0 if row.inefficency == None else row.inefficency

                # controlliamo che il campo note e note inefficienze rispettino i parametri di lunghezza
                if len(row.note_commessa) > 972:
                    # 972 perchè dobbiamo concatenare alla fine delle note la frase: '.Ore inserite tramite AI'
                    logger.debug(f"il campo note per l intervento {row.codice_intervento} a superato i caratteri massimi, si dovrà troncare una parte delle note per continuare.")
                    row.note_commessa = row.note_commessa[:970]
                    
                if len(row.inefficency_note) > 200:
                    logger.debug(f"il campo note inefficienza  per l intervento {row.codice_intervento} a superato i caratteri massimi, si dovrà troncare una parte delle note per continuare.")
                    row.note_commessa = row.note_commessa[:199]

                # aggiungiamo il placeholder
                row.note_commessa = row.note_commessa + ".Ore inserite tramite AI"
                logger.debug(f"per l intervento {row.codice_intervento} da salvare su sql server la nota è stata aggiornata nel seguente modo: {row.note_commessa}")

                # rimuoviamo la timezone della data_intervento 2026-01-28 00:00:00+00:50
                data_intervento_mod = row.data_intervento.isoformat()
                data_intervento_split = data_intervento_mod.split('+')
                # prendiamo solo la prima parte
                data_intervento_mod = data_intervento_split[0]
                logger.debug(f"data intervento modificata da {row.data_intervento} -> {data_intervento_mod}")

                # call the procedure
                query = f"EXEC SP_UPD0001_Aggiornamento_righe_intervento  0, '{row.codice_intervento}', '{row.matricola_tecnico}', '{data_intervento_mod}', {row.ore_ordinarie}, {row.ore_straordinarie}, {row.ore_viaggio},{row.note_commessa}, {row.inefficency}, {row.inefficency_type}, {row.inefficency_note}"
                logger.debug(f"query per fare l'update: {query}")
                cursor.execute(query)
                # una PRINT in una stored procedure mostra i risulati in un campo messages:
                if cursor.messages:
                    ris = cursor.messages[0][1] # [0] il messaggio della exec, [1] testo
                    logger.info(f"Il risultato dal SQLServer è: {ris}")

            except Exception as e:
                logger.error(f"Errore nella salvare la commessa {row.commessa}: {e}")
                return False
        else:
            # use csv 
            logger.debug(f"commessa: {row}")
            self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'ORE_ORDINARIE'] = row.ore_ordinarie 
            self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'ORE_STRAORDINARIE'] = row.ore_straordinarie 
            self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'ORE_VIAGGIO'] = row.ore_viaggio

            if row.find_inefficienza == True:
                self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'INEFFICIENCY'] = row.inefficency
                self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'INEFFICIENCY_TYPE'] = row.inefficency_type
                self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'HOURS_BY_CLIENT'] = row.inefficency_note

            self.data.loc[self.data['COMMESSA'].astype(str) == row.commessa, 'NOTE'] = row.note_commessa

            logger.debug(f"dati aggiornati: {self.data[['ORE_ORDINARIE','ORE_STRAORDINARIE', 'ORE_VIAGGIO', 'INEFFICIENCY','NOTE']]}")

            # salviamo nel file
            self.data.to_csv(CSV_FILENAME_FOLLOW)
        return True 

    def __init__(self, config: DictConfig):
        self.config = config
        self.data = self._upload_data()

    def  is_empty(self):
        return len(self.data) == 0    
    
def process_cel(cell_number):
    """
    Processing della stringa telefono. Bisogna rimuovere il + davanti e gli spazi
    """
    tmp = cell_number.strip()
    
    tmp = tmp[1:] if tmp[0] == '+' else tmp
    return tmp.replace(" ", "")

class DB_Postgres:
    """
    Classe per interfacciarsi con Postgressql
    """
    def __init__(self, config:DictConfig):
        self.config = config
        self.tables = []
        self.columns_name = {}
              
    def convert2DF(self, results, table, new_columns:list[str]=None)-> pd.DataFrame:
        """
            processa i risultati del database e li fornisce come df
            results: i dati da mostrare 
            table: il nome della tabella per ottenere gli indici di colonna
            new_columns: lista di nomi di colonne da aggiungere
        """
        data = None
        if new_columns:
            columns_old = self.columns_name[table]
            for column in new_columns:
                # inseriamo solo le colonne non presenti già
                if column not in columns_old:
                    columns_old.append(column)

            data = pd.DataFrame(results, columns=columns_old)
        else:
            data = pd.DataFrame(results, columns=self.columns_name[table])

        return data
    
    def drop_table(self, table:str):
        """
        Drop della tabella selezionata 
        """
        assert table in self.tables
        conn = psy.connect(self.conn_credentials)
        for table_name in self.tables:
            if table_name == table:
                query = psy.sql.SQL("DROP TABLE IF EXISTS {}").format(psy.sql.Identifier(table_name))
                conn.execute(query)
        
        # salvo le modifiche
        conn.commit()
        logger.debug(f"la tabella {table} è state eliminate")
        # tolgo al tabella dalla lista 
        self.tables.remove(table)
        # e le sue colonne
        self.columns_name.pop(table)
        
    def add_table(self, table_name:str, schema:str, column_names:list[str]):
        """
        Aggiunta di una tabella nella classe DB Postgres, se non esiste la tabella, la crea
        - table_name(str): nome della tabella
        - schema(str): lo schema da utilizzare per creare la tabella
        - columns(list[str]): lista dei nomi di colonna della tabella
        """
        with psy.connect(self.conn_credentials) as conn:
            # controlliamo se esiste la tabella
            try:
                # non si può fare il binding di nomi di tabelle e colonne su psycopg 3(perchè li compila separatamente poi sul server), bisogna fare il binding su client 
                conn.execute(psy.sql.SQL("SELECT * FROM {}").format(psy.sql.Identifier(table_name)))
                logger.debug(f"la tabella {table_name} è gia presente")
            except psy.errors.UndefinedTable as e:
                conn.rollback()
                
                # creiamo la tabella
                query = psy.sql.SQL("CREATE TABLE {} ({})").format(psy.sql.Identifier(table_name), psy.sql.SQL(schema))
                conn.execute(query)
                logger.debug(f"creazione della tabella {table_name}")

             
        # aggiungiamo la tabella alla lista
        self.tables.append(table_name)
        # aggiungiamo le colonne della tabella
        self.columns_name[table_name] = column_names

    def insert_row(self, table_name:str, output_h:str, output_v:str)->bool:
        """
        Funzione per inserire una riga in una tabella
        table_name(str): il nome della tabella
        output_h(str): stringa che contiene tutti gli header della tabella 
        output_v(str): stringa che contiene tuti i valori della tabella
        """
        try:
            with psy.connect(self.conn_credentials) as conn:
                query = psy.sql.SQL("INSERT INTO {} {} VALUES {}").format(psy.sql.Identifier(table_name), psy.sql.SQL(output_h), psy.sql.SQL(output_v))
                logger.debug(f"riga da aggiungere alla tabella: {query.as_string()}")
                ris = conn.execute(query)
                return True
        except Exception as e:
            logger.error(f"Errore nella scrittura sul db {self.db_name}: {str(e)}")
            return False

    @abstractmethod
    def insert(self, table_name:str, row:SQLModel)->bool:
        """
        Funzione astratta per aggiungere una riga ad una tabella 
        """
        pass
    
    def export_to_sql_server(self, table_name:str)->bool:
        """Funzione per esportare tutti i dati del db in una tabella Sql server"""
        all_data = self.select_all(table_name)
        if all_data.empty:
            return False
        try:
            conn = pyodbc.connect(self.config.database.sqlserver_conn, autocommit=True)
            cursor = conn.cursor()
            for _,row in  all_data.iterrows():
                logger.debug(f"row: {row}")
                if self.config.app.name == 'tecnico':
                    mittente = row['matricola_tecnico'] if row['cel_mittente'] != self.config.whatsapp.phone_id else "CED"
                    destinatario = row['matricola_tecnico'] if row['cel_destinatario'] != self.config.whatsapp.phone_id else "CED"
                    testo = row['testo'] if row['testo'] != None else "messaggio vuoto"
                    riferimento = row['codice_intervento']
                else:
                    mittente = row['referente'] if row['cel_mittente'] != self.config.whatsapp.phone_id else "CED"
                    destinatario = row['referente'] if row['cel_destinatario'] != self.config.whatsapp.phone_id else "CED"
                    testo = row['testo'] if row['testo'] != None else "messaggio vuoto"
                    riferimento = row['cod_offerta']

                    
                    

                chat_id = row['cel_destinatario'] if row['cel_destinatario'] != self.config.whatsapp.phone_id else row['cel_mittente']
                # remove milliseconds
                try:
                    date = datetime.strptime(str(row['last_modify']),'%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%dT%H:%M:%S")
                except Exception as e:
                    # dovrebbe essere un problema nel format che adesso é .%f
                    date = datetime.strptime(str(row['last_modify']),'%Y-%m-%d %H:%M:%S.%f').strftime("%Y-%m-%dT%H:%M:%S")
                
                query ="""
                        INSERT INTO [dbo].[AIFT_CHAT_LOG_WA]
                              ([CHAT_ID]
                                ,[CHAT_ORDINE]
                                ,[CHAT_DATA]
                                ,[CHAT_RIFERIMENTO]
                                ,[CHAT_CATEGORIA]
                                ,[CHAT_MITTENTE]
                                ,[CHAT_DESTINATARIO]
                                ,[CHAT_MESSAGGIO]
                                ,[CHAT_STATO]
                                ,[CHAT_AI_COMPLETE])
                        VALUES
                              ('{}'
                              ,{}
                              ,'{}'
                              ,'{}'
                              ,'{}'
                              ,'{}'
                              ,'{}'
                              ,'{}'
                              ,'{}'
                              ,{})
                        """.format(chat_id,row['cronologia'], date, riferimento, self.config.app.name, mittente, destinatario, testo, row['status'], int(row['complete']))
                logger.debug(query)
                cursor.execute(query)
                cursor.commit()
            logger.info(f"tutti le righe della tabella {table_name} sono state inviate al server SQL")

        except Exception as e:
            logger.error(f"errore nell'esportare i dati nel server SQL per la tabella {table_name}: {e}")
            return False

    def export_to_csv(self, table_name:str, output_folder:str="./history/") ->bool:
        """Funzione per esportare tutti i dati del db in un file csv"""
        all_data = self.select_all(table_name)
        if all_data.empty:
            return False
        try:
            filename = f"session_{datetime.now(tz=timezone('Europe/Rome')).strftime("%d_%m_%Y_%H_%M")}.csv"
            output_path = os.path.join(output_folder,filename)

            all_data.to_csv(output_path, index_label=False)
        except Exception as e:
            logger.error(f"errore nell'esportare i dati della tabella {table_name}: {e}")
            return False
        
        logger.info(f"salvato la cronologia dei messaggi del db nel file: {output_path}")
        return True
    
    def select_all(self, table_name:str)-> pd.DataFrame|None:
        """
        Funzione per ritornare tutte le righe della tabella
        """
        assert table_name in self.tables
        with psy.connect(self.conn_credentials) as conn:
            try:
                results = conn.execute(psy.sql.SQL("SELECT * FROM {}").format(psy.sql.Identifier(table_name))).fetchall()

                results_df = self.convert2DF(results,table_name)
            except Exception as e:
                logger.error(f"Errore nella lettura del db {self.db_name} nella tabella {table_name}: {str(e)}")
                return None
        return results_df

    def delete(self,table_name, where_field:str, where_value:any) -> bool:
        """
        Funzione che elimina una riga dato un campo:
        - table_name(str): nome della tabella
        - where_field(str): che campo cercare
        - where_value(any): il valore del campo da cercare
        """
        assert table_name in self.tables
        try:
            logger.debug(f"dentro al delete della tabella {table_name}")
            # costruiamo la query
            query = psy.sql.SQL("DELETE FROM {} WHERE {} = {};").format(psy.sql.Identifier(table_name),psy.sql.Identifier(where_field), psy.sql.Literal(where_value))
            
            logger.warning(query.as_string())
            with psy.connect(self.conn_credentials) as conn:
                ris = conn.execute(query)
                logger.debug(f"riga eliminata nella tabella: {query.as_string()}")
            return True
        except Exception as err:
            logger.error(f"Errore nel delete del db {self.db_name} nella tabella {table_name}: {str(err)}")
            return False

    def find(self, table_name:str, by:str , value:any, limit:int=None, order_field:str=None, order_direction:str=None)-> pd.DataFrame|None:
        """
        Funzione per ritornare delle righe avendo un campo in ingresso
        - table_name(str): nome della tabella
        - by(str): che campo cercare
        - value(any): il valore del campo 
        - limit(int): numero di elementi ritornati massimo se None ritorna tutti
        - order_field(str): il campo su cui effettuare l'ordinamento
        - order_direction(str): la direzione dell'ordinamento
        """
        assert table_name in self.tables

        # costruiamo la query
        query = psy.sql.SQL("SELECT * FROM {} WHERE {} = {}").format(psy.sql.Identifier(table_name),psy.sql.Identifier(by), psy.sql.Literal(value))
        if order_field:
            query = psy.sql.SQL("SELECT * FROM {} WHERE {} = {} ORDER BY {} {}").format(psy.sql.Identifier(table_name),psy.sql.Identifier(by), psy.sql.Literal(value), psy.sql.Identifier(order_field), psy.sql.SQL(order_direction))
        
        
        with psy.connect(self.conn_credentials) as conn:
            try:
                results = conn.execute(query=query).fetchall()

                results_df = self.convert2DF(results, table_name)
            except Exception as e:
                logger.error(f"Errore nel fare la search nel db {self.db_name} : {str(e)}")
                return None

        return results_df

    def update(self, table_name, where_field:str, where_value:any, set_field:str, set_value:any) -> bool:
        """
        Funzione che aggiorna una riga dato un campo:
        - table_name(str): nome della tabella
        - where_field(str): che campo cercare
        - where_value(any): il valore del campo da cercare
        - set_field(str): il campo da cambiare 
        - set_value(any): il valore da inserire
        """
        assert table_name in self.tables
        try:
            logger.debug(f"dentro all'update della tabella {table_name}")
            # costruiamo la query
            query = psy.sql.SQL("UPDATE {} SET {} = {} WHERE {} = {};").format(psy.sql.Identifier(table_name),psy.sql.Identifier(set_field), psy.sql.Literal(set_value),
                                                                                psy.sql.Identifier(where_field), psy.sql.Literal(where_value))
            
            logger.debug(query.as_string())
            with psy.connect(self.conn_credentials) as conn:
                ris = conn.execute(query)
                logger.debug(f"riga aggiornata nella tabella: {query.as_string()}")
            return True
        except Exception as err:
            logger.error(f"Errore nel update del db {self.db_name} nella tabella {table_name}: {str(err)}")
            return False



class DB_Messaggi(DB_Postgres):
    """Classe per il db dei messaggi di Postgres"""


    def _fake_obj(self, app:str) -> SQLModel|None:
        """
        Funzione per creare oggetti Messaggi fittizzi
        """
        match app:
            case 'tecnico':
                # ritorniamo un oggetto fittizzio per app tecnici
                return MessaggioTecnici(ruolo='user', commessa="27384", cel_mittente= "346784354", matricola_tecnico="ciao", data_intervento=datetime.now(tz=timezone('Europe/Rome')),codice_intervento="12010",cel_destinatario="3468523354", msg_id="prova")
            case 'follow':
                # ritorniamo un oggetto fittizzio per app follow up
                return MessaggioFollowUp(ruolo="user", cod_offerta="270145", data_ultimo_contatto=datetime.now(tz=timezone('Europe/Rome')),titolo="ciao", cliente="sapa")
            case _:
                logger.error(f"Non è stato possibile creare un messaggio fittizzio perchè non corrisponde a nessuna app. nome dell app passata -> {app} possibile nomi sono 'tecnico' e 'follow'")
                return None
    def _get_schema(self) -> str:
        """
        Funzione per ritornare lo schema della tabella in relazione a che app si utilizza
        """
        if isinstance(self.obj, MessaggioTecnici):
            # stiamo utilizzando i tecnici per la seguente tabella
            conn = psy.connect(self.conn_credentials)
            return  self.obj.toQuery(primarykey=['msg_id'], conn=conn, vec_dim=self.config.database.vector_dim, isnotnull=['cronologia', 'cel_mittente', 'cel_destinatario','commessa','ruolo','codice_intervento','matricola_tecnico'])
        else:
            # utilizziamo i commerciali per la seguente tabella
            conn = psy.connect(self.conn_credentials)
            return  self.obj.toQuery(primarykey=['msg_id'], conn=conn, double_precision=['prezzo_vendita'], isnotnull=['cronologia', 'cel_mittente', 'cel_destinatario','cod_offerta','ruolo','data_ultimo_contatto','titolo','cliente'])

    def __init__(self, config, app:str, is_scheduler:bool=False):
        """Metodo per inizializzare la classe:
         - app(str): può essere tecnico/follow per intendere a che oggetto SQLModel fare riferimento per creare la tabella
         - is_scheduler(bool): booleano che dice se si tratta di un processo scheduler oppure no
        """
        super().__init__(config)
        # aggiungiamo la tabella dei messagi
        self.db_name = self.config.database.postgres_db_name_msg
        table_msg = self.config.database.postgres_msg_table

        # aggiungiamo le credenziali per la connessione 
        self.conn_credentials = self.config.database.postgres_msg_conn 
        # creiamo un Messaggio fittizio
        self.obj = self._fake_obj(app)
        if self.obj is None:
            logger.error(f"Errore nella creazione del messaggio fittizzio, la connessione al db non verrà fatta")
            return 
        
        # create the connection with postgres
        conn = psy.connect(self.config.database.postgres_msg_conn)
        # aggiungiamo l'estensione per gli embeddings
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        logger.info(f"Connessione al db {self.db_name} Effettuata")
        
        # Ogni volta che si avvia l'app facciamo il drop della tabella messaggi in fase di development 
        if config.info.env == "TEST":
            # aggiunta della tabella nella classe
            self.tables.append(table_msg)
            self.columns_name = {table_msg: self.obj.columns()}
            self.drop_table(table=table_msg)
            # aggiungiamo la tabella Messaggi
            self.add_table(table_msg)
        else:
            # il drop della tabella viene gestito a livello di docker, quindi che sia scheduler o executor si deve SOLO aggiungere la tabella
            self.add_table(table_msg)
            


    def add_table(self, table_name:str):
        """
        Funzione che aggiunge una tabella al db
        
        :param table_name: nome della tabella
        """
        # prendiamo lo schema dell'elemento fittizzio
        schema = self._get_schema()
        #logger.debug(f"schema da passare: {schema}")
        return super().add_table(table_name, schema, column_names=self.obj.columns())

    def insert(self, table_name, row):
        """
        Funzione per aggiungere una riga ad una tabella 
        """
        assert table_name in self.tables

        try:
            # fetch the data
            d= row.model_dump(exclude_none=True)

            # prendi gli headers e i valori
            headers,values = zip(*d.items())

            # preparo la stringa di output
            output_h = "("
            output_v = "("
            for idx,header in enumerate(headers):
                if idx != len(headers) - 1:
                    output_h += f"{header}, "
                    if header == 'ruolo':
                        # dobbiamo fornigli il value. Es: RuoloEnum.USER -> user
                        output_v += f"'{values[idx].value}', "
                    else:
                        # cotrolliamo che  commesse non sia nullo
                        if values[idx] == '' and header == 'commessa':
                            raise Exception('la commessa non può essere nulla')
                        else:
                            output_v += f"'{values[idx]}', "
                else:
                    output_h += f"{header})"
                    output_v += f"'{values[idx]}')"
            
            self.insert_row(table_name, output_h, output_v)
        except Exception as e:
            logger.error(f"Errore nell'inserimento della riga per il db {self.db_name} e la taballa {table_name}: {str(e)}")
            return False
        return True

        

class DB_Vector(DB_Postgres):
    """Classe per il vector db su Postgres"""

    # Dict per le funzioni di similarità
    similarity_methods = {
            'cosine': '<=>',
            'l2': '<->',
            'inner': '<#>',
            'l1': '<+>'
    }
    
    def __init__(self, config):
        super().__init__(config)
        # aggiungiamo la tabella dei messagi
        self.db_name = self.config.database.postgres_db_name_vector
        table_name = self.config.database.postgres_vc_table
        self.tables.append(table_name)

        # aggiungiamo le credenziali per la connessione 
        self.conn_credentials = self.config.database.postgres_vectordb_conn
        try:
            conn = psy.connect(self.conn_credentials)
            # aggiungiamo l'estensione per gli embeddings
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            logger.info(f"Connessione al db {self.db_name} Effettuata")
        except Exception as e:
            logger.error(f"Errore nella connessione al db: {self.db_name} con password: {self.config.database.postgres_password}: {e}")

    def insert(self, table_name, row):
        """
        Funzione per aggiungere una riga ad una tabella 
        """
        assert table_name in self.tables

        

        try:
            # fetch the data
            d= row.model_dump(exclude_none=True)

            # prendi gli headers e i valori
            headers,values = zip(*d.items())

            # preparo la stringa di output
            output_h = "("
            output_v = "("
            for idx,header in enumerate(headers):
                if idx != len(headers) - 1:
                    output_h += f"{header}, "
                    # cotrolliamo che  commesse non sia nullo
                    if values[idx] == '' and header == 'embedding':
                        raise Exception('embedding non può essere nulla')
                    else:
                        output_v += f"'{values[idx]}', "
                else:
                    output_h += f"{header})"
                    output_v += f"'{values[idx]}')"
            
            self.insert_row(table_name, output_h, output_v)
        except Exception as e:
            logger.error(f"Errore nell'inserimento della riga per il db {self.db_name} e la taballa {table_name}: {str(e)}")
        
    def add_table(self, table_name):
        # creiamo un elemento fittizio
        etc = Etichetta(etichetta="prova", testo="questa è una prova", embedding=[0.14,0.11])
        schema_str = etc.toQuery(primarykey=['embedding'], vec_dim=self.config.database.vector_dim, isnotnull=['etichetta', 'testo'])
        return super().add_table(table_name, schema_str, column_names=etc.columns())
    
    def find_similar(self, table_name:str, query_emb:list[float], method:str='cosine', topk:int=5, column_name:str='embedding'):
        """
        Funzione per trovare vettori simili
        - query_emb(list[float]): embedding della query
        - method(str): che metodo utilizzare per fare il confronto tra vettori
        - topk(int): quanti elementi ritornare 
        - column_name(str): colonna in cui cercare
        """
        assert table_name in self.tables
    
        func = self.similarity_methods[method]
        # costruiamo la query
        query = psy.sql.SQL("SELECT *,{} {} {} AS distance FROM {} ORDER BY distance LIMIT {} ").format(psy.sql.Identifier(column_name), psy.sql.SQL(func),
                                                                                                psy.sql.Literal(str(query_emb)), psy.sql.Identifier(table_name), psy.sql.Literal(topk))
        logger.debug(f"ricerca per etichetta con query: {query.as_string()}")
        with psy.connect(self.conn_credentials) as conn:
            try:
                results = conn.execute(query=query).fetchall()
                results_df = self.convert2DF(results, table_name, new_columns=["distance"])
                
            except Exception as e:
                logger.error(f"Errore nel fare la search nel db {self.db_name} : {str(e)}")
                return None

        return results_df

