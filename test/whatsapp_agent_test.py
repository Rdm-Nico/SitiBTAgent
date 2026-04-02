import unittest
from unittest.mock import patch, MagicMock, ANY
from flask import Flask
from omegaconf import OmegaConf
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from pytz import timezone
import sys

# Aggiunge la directory root del progetto al path di Python per trovare i moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from whatsapp_agent_tecnico import Whatsapp_Agent 
from utils.Event import Event
from schema.structured_ouput_model import Tecnico

class TestWhatsappAgent(unittest.TestCase):

    @patch('whatsapp_agent.ConcurrentThreadPoolExecutor')
    @patch('whatsapp_agent.BackgroundScheduler')
    @patch('whatsapp_agent.clientRouter')
    @patch('whatsapp_agent.DB_Vector')
    @patch('whatsapp_agent.DB_Messaggi')
    @patch('whatsapp_agent.DB_Commesse')
    def setUp(self, MockDBCommesse, MockDBMessaggi, MockDBVector, 
              MockClientRouter, MockScheduler, MockExecutor):
        """
        Setup eseguito prima di ogni test.
        Inizializza una configurazione fittizia e l'istanza di Whatsapp_Agent con dipendenze mockate.
        """
        self.config = OmegaConf.create({
            'whatsapp': {
                'port': 5000,
                'access_token': 'test_token',
                'phone_id': '123456789',
                'url': 'http://test.whatsapp.url',
                'version': 'v18.0',
                'verify_token': 'test_verify_token'
            },
            'database': {
                'postgres_msg_table': 'commesse',
                'postgres_db_name_msg': 'testdb',
                'is_db': False,
                'postgres_db_name_vector': 'vectordb',
                'postgres_vc_table': 'etichette',
                'postgres_vectordb_conn': 'host=localhost port=5432 dbname=test',
                'postgres_msg_conn': 'host=localhost port=5432 dbname=test',
                'vector_dim': 128
            },
            'whisper': {
                'model': 'tiny'
            },
            'info': {
                'env': 'TEST',
                'save_chat': False
            }
        })
        
        # Configura i mock PRIMA di creare l'agent
        self.mock_db_commesse = MockDBCommesse.return_value
        self.mock_db_commesse.is_empty.return_value = True  # Evita l'avvio automatico
        self.mock_db_commesse.data = pd.DataFrame()  # DataFrame vuoto
        
        self.mock_db_messaggi = MockDBMessaggi.return_value
        self.mock_db_vector = MockDBVector.return_value
        self.mock_scheduler = MockScheduler.return_value
        self.mock_executor = MockExecutor.return_value
        
        # Mock dei client
        self.mock_client = MagicMock()
        MockClientRouter.return_value = self.mock_client
        
        # Ora creiamo l'agent
        self.agent = Whatsapp_Agent(self.config)
        
        # Sostituiamo il logger con un mock
        self.agent.logger = MagicMock()
        
        # Riferimento diretto al mock del message_db per i test
        self.agent.message_db = self.mock_db_messaggi


    def _create_event_data_failed(self, sender="391234567890", msg_id="test_msg_123"):
        """Crea dati per un evento di tipo 'failed'"""
        return {
            "entry": [{
                "changes": [{
                    "value": {
                        "statuses": [{
                            "id": msg_id,
                            "status": "failed",
                            "recipient_id": sender,
                            "timestamp": "1234567890"
                        }]
                    }
                }]
            }]
        }
    
    def _create_event_data_message(self, sender="391234567890", msg_id="test_msg_123", body="Test message"):
        """Crea dati per un messaggio in arrivo"""
        return {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": msg_id,
                            "from": sender,
                            "type": "text",
                            "text": {"body": body},
                            "timestamp": "1234567890"
                        }]
                    }
                }]
            }]
        }
    
    def _create_commessa_dataframe(self, failed_count=0):
        """Crea un DataFrame che simula il risultato di message_db.find()"""
        TIMEZONE = timezone('Europe/Rome')
        return pd.DataFrame([{
            'commessa': 'COM001',
            'matricola_tecnico': 'TEC001',
            'data_intervento': datetime.now(tz=TIMEZONE),
            'codice_intervento': 'INT001',
            'cronologia': 0,
            'body': 'Messaggio di test',
            'cel_destinatario': '391234567890',
            'ruolo': 'assistant',
            'cel_mittente': '123456789',
            'embedding': [],  # Lista vuota invece di None
            'complete': False,
            'ore_ordinarie': 0.0,
            'ore_straordinarie': 0.0,
            'ore_viaggio': 0.0,
            'find_inefficienza': False,
            'inefficency': 0.0,
            'inefficency_type': '',  # Stringa vuota invece di None
            'inefficency_note': '',  # Stringa vuota invece di None
            'note_commessa': '',  # Stringa vuota invece di None
            'tool_name': '',  # Stringa vuota invece di None
            'failed_count': failed_count
        }])

    def test_initialization(self):
        """
        Testa che l'agente sia inizializzato correttamente.
        """
        self.assertIsInstance(self.agent.app, Flask)
        self.mock_chat_client.add_model.assert_any_call('test_chat_model')
        self.mock_extractor_client.add_model.assert_any_call('test_extractor_model')
        self.mock_translate_client.add_model.assert_any_call('test_translate_model')
        self.mock_scheduler.start.assert_called_once()
        self.assertIsNotNone(self.agent.db)
        self.assertIsNotNone(self.agent.message_db)

    def test_create_template(self):
        """
        Testa la creazione del template per il messaggio.
        """
        row = {
            'COMMESSA': '12345',
            'DATA_INTERVENTO': datetime(2023, 10, 27),
            'DES_CLIENTE': 'Cliente Test S.R.L. con nome lungo',
            'MATR': 'M123',
            'CODICE_INTERVENTO': 'INT001',
            'TECNICO_MOBILE': '3331234567'
        }
        template = self.agent._create_template(row)
        self.assertEqual(template['name'], 'template_attivita_tecnico')
        self.assertEqual(template['language_code'], 'it')
        self.assertEqual(template['commessa'], '12345')
        self.assertEqual(template['cliente'], 'Cliente Test S.R.L.') # Test strip e taglio a 20 caratteri


    def test_create_template_failure(self):
        """
        Testa la creazione del template per il messaggio con un numer non valido
        """
        row = {
            'COMMESSA': '12345',
            'DATA_INTERVENTO': datetime(2023, 10, 27),
            'DES_CLIENTE': 'Cliente Test S.R.L. con nome lungo',
            'MATR': 'M123',
            'CODICE_INTERVENTO': 'INT001',
            'TECNICO_MOBILE': ''
        }
        template = self.agent._create_template(row)
        self.assertEqual(template['errore'],'numero non presente')

    def test_add_job_now(self):
        """
        Testa l'aggiunta di un job da eseguire immediatamente.
        """
        row = {
            'COMMESSA': '12345', 'DATA_INTERVENTO': datetime.now(), 'DES_CLIENTE': 'Test',
            'MATR': 'M123', 'CODICE_INTERVENTO': 'INT001', 'TECNICO_MOBILE': '3331234567'
        }
        self.agent.add_job(row, when=0)
        self.mock_scheduler.add_job.assert_called_once_with(
            self.agent.send_message,
            kwargs={'to': '3331234567', 'template': ANY},
            id='process_12345',
            replace_existing=True
        )

    def test_add_job_scheduled(self):
        """
        Testa l'aggiunta di un job schedulato per il futuro.
        """
        row = {
            'COMMESSA': '54321', 'DATA_INTERVENTO': datetime.now(), 'DES_CLIENTE': 'Test',
            'MATR': 'M321', 'CODICE_INTERVENTO': 'INT002', 'TECNICO_MOBILE': '3337654321'
        }
        self.agent.add_job(row, when=1) # 1 ora
        self.mock_scheduler.add_job.assert_called_once()
        args, kwargs = self.mock_scheduler.add_job.call_args
        self.assertEqual(args[1], 'date')
        self.assertIn('run_date', kwargs)
        self.assertAlmostEqual(kwargs['run_date'].timestamp(), (datetime.now() + timedelta(hours=1)).timestamp(), delta=5)

    @patch('whatsapp_agent.Thread')
    @patch('whatsapp_agent.time')
    def test_start_with_data(self, mock_time, MockThread):
        """
        Testa l'avvio dell'agente quando ci sono dati da processare.
        """
        self.mock_db_commesse.is_empty.return_value = False
        mock_data = pd.DataFrame([
            {'MATR': 'M1', 'COMMESSA': 'C1', 'DATA_INTERVENTO': datetime.now(), 'DES_CLIENTE': 'Cli1', 'CODICE_INTERVENTO': 'I1', 'TECNICO_MOBILE': '111'},
            {'MATR': 'M1', 'COMMESSA': 'C2', 'DATA_INTERVENTO': datetime.now(), 'DES_CLIENTE': 'Cli2', 'CODICE_INTERVENTO': 'I2', 'TECNICO_MOBILE': '111'},
            {'MATR': 'M2', 'COMMESSA': 'C3', 'DATA_INTERVENTO': datetime.now(), 'DES_CLIENTE': 'Cli3', 'CODICE_INTERVENTO': 'I3', 'TECNICO_MOBILE': '222'},
        ])
        self.agent.db.data = mock_data

        # Sostituisce il metodo reale con un MagicMock per poter tracciare le chiamate
        self.agent.add_job = MagicMock()
        
        self.agent.start()

        MockThread.assert_called_once_with(target=self.agent._run)
        self.assertEqual(self.agent.add_job.call_count, 3)
        
        # Verifica che add_job sia stato chiamato con gli argomenti corretti.
        calls = self.agent.add_job.call_args_list
        
        # La logica di start() dovrebbe chiamare add_job in ordine:
        # 1. Commessa C1 (primo lavoro per M1) -> when=0
        # 2. Commessa C2 (secondo lavoro per M1) -> when=6
        # 3. Commessa C3 (primo lavoro per M2) -> when=0
        
        # Verifica la chiamata per la commessa C1 (when=0)
        self.assertEqual(calls[0].args[0]['COMMESSA'], 'C1')
        self.assertEqual(calls[0].args[1], 0) # Ecco come controlli il 'when'
        # Verifica la chiamata per la commessa C2 (when=6)
        self.assertEqual(calls[1].args[0]['COMMESSA'], 'C2')
        self.assertEqual(calls[1].args[1], 6)
        # Verifica la chiamata per la commessa C3 (when=0)
        self.assertEqual(calls[2].args[0]['COMMESSA'], 'C3')
        self.assertEqual(calls[2].args[1], 0)
        

    @patch('whatsapp_agent.Thread')
    def test_start_with_no_data(self, MockThread):
        """
        Testa che l'agente non si avvii se non ci sono dati.
        """
        self.mock_db_commesse.is_empty.return_value = True
        self.agent.start()
        MockThread.assert_not_called()
        self.agent.logger.info.assert_called_with("Non sono presenti righe da processare per oggi. L'App whatsapp non si avvierà")


    @patch('requests.post')
    def test_send_message_text_it(self, mock_post):
        """
        Testa l'invio di un semplice messaggio di testo in italiano.
        """
        mock_post.return_value.json.return_value = {'messages': [{'id': 'wamid.test.text'}]}
        mock_post.return_value.status_code = 200
        # Mockiamo il metodo save2DB per assicurarci che NON venga chiamato
        self.agent.save2DB = MagicMock()

        self.agent.send_message(to='393468413354', message='Ciao')
        # controlliamo che non chiami la translate perchè il numero è italiano 
        

        mock_post.assert_called_once()
        sent_data = mock_post.call_args.kwargs['json']
        self.assertEqual(sent_data['type'], 'text')
        self.assertEqual(sent_data['text']['body'], 'Ciao')
        # Verifichiamo che save2DB NON sia stato chiamato, come da logica del codice
        self.agent.save2DB.assert_not_called()

    @patch('requests.post')
    def test_send_message_text_en(self, mock_post):
        """
        Testa l'invio di un semplice messaggio di testo in inglese.
        """
        mock_post.return_value.json.return_value = {'messages': [{'id': 'wamid.test.text'}]}
        mock_post.return_value.status_code = 200
        # Mockiamo il metodo save2DB per assicurarci che NON venga chiamato
        self.agent.save2DB = MagicMock()
        # aggiungiamo il mock translate text
        self.mock_translate_client.translate_text.return_value = {
            'response': 'Hello'
        }

        self.agent.send_message(to='+20 100 285 9921', message='Ciao') 
        

        mock_post.assert_called_once()
        sent_data = mock_post.call_args.kwargs['json']
        self.assertEqual(sent_data['type'], 'text')
        self.assertEqual(sent_data['text']['body'], 'Hello')
        # Verifichiamo che save2DB NON sia stato chiamato, come da logica del codice
        self.agent.save2DB.assert_not_called()

    def test_useExtractorModel_success(self):
        """
        Testa il modello estrattore con una risposta JSON valida.
        """
        json_response = '{"ore_ordinarie":6.0,"ore_straordinarie":0.0,"ore_viaggio":0.0,"durata_inefficienza":0.0, "note_inefficienza": "", "note":"commessa 230449 è sbagliata","commessa":"errors"}'
        self.mock_extractor_client.generate.return_value = {'response': f"```json\n{json_response}\n```"}
        
        result = self.agent.useExtractorModel(input="some text")

        print(result)
        self.assertIsInstance(result, Tecnico)
        self.assertEqual(result.ore_ordinarie, 6.0)
        self.assertEqual(result.durata_inefficienza, 0.0)
        self.assertEqual(result.note, "commessa 230449 è sbagliata")

    def test_useExtractorModel_failure(self):
        """
        Testa il modello estrattore quando restituisce un JSON malformato.
        """
        self.mock_extractor_client.generate.return_value = {'response': "testo invalido non json"}
        
        # Il metodo dovrebbe catturare l'eccezione e loggarla, ritornando None o un oggetto vuoto
        # a seconda dell'implementazione del try-except. Qui assumiamo che logghi e non rilanci.
        predict = self.agent.useExtractorModel(input="some text")
        # A seconda della gestione dell'errore, `predict` potrebbe essere None o un'istanza di default
        # La tua implementazione corrente non ritorna nulla in caso di errore, quindi `predict` sarà None
        self.assertIsNone(predict)

    def test_checkLevel_single_new_commessa(self):
        """
        Testa checkLevel per un tecnico con una sola commessa, al primo messaggio.
        """
        mock_df = pd.DataFrame([{
            'commessa': 'C1', 'cronologia': 0, 'complete': False, 
            'matricola_tecnico': 'M1', 'data_intervento': datetime.now(), 'codice_intervento': 'I1'
        }])
        self.mock_db_postgres.find.side_effect = [
            pd.DataFrame(columns=mock_df.columns), # Prima chiamata (by cel_mittente) non trova nulla
            mock_df         # Seconda chiamata (by cel_destinatario) trova la commessa iniziale
        ]
        
        msg_recived = MagicMock(spec=Event, sender='111')
        result_df = self.agent.checkLevel(msg_recived)

        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]['commessa'], 'C1')
        self.assertEqual(result_df.iloc[0]['cronologia'], 1) # Cronologia incrementata

    def test_checkLevel_multiple_commesse(self):
        """
        Testa checkLevel per un tecnico con più commesse attive.
        """
        mock_df = pd.DataFrame([
            {'ruolo': 'assistant','commessa': 'C1', 'cronologia': 0, 'complete': False, 'matricola_tecnico': 'M1', 'data_intervento': datetime.now(), 'codice_intervento': 'I1'},
            {'ruolo': 'assistant','commessa': 'C2', 'cronologia': 0, 'complete': False, 'matricola_tecnico': 'M1', 'data_intervento': datetime.now(), 'codice_intervento': 'I2'}
        ])
        self.mock_db_postgres.find.return_value = mock_df
        
        msg_recived = MagicMock(spec=Event, sender='111')
        result_df = self.agent.checkLevel(msg_recived)

        self.assertEqual(len(result_df), 2)
        self.assertTrue(all(result_df['cronologia'] == 2)) # 0 + 1 (base) + 1 (plus)

    @patch('whatsapp_agent.Whatsapp_Agent.analyzeMessage')
    def test_manageRecivedMessage_single_commessa(self, mock_analyze):
        """
        Testa che `analyzeMessage` venga chiamato per una commessa singola.
        """
        commessa_df = pd.DataFrame([{'commessa': 'C1', 'cronologia': 1, 'matricola_tecnico': 'M1'}])
        commessa_df.to_dict = MagicMock(return_value=[{'commessa': 'C1', 'cronologia': 1, 'matricola_tecnico': 'M1'}])
        
        self.agent.checkLevel = MagicMock(return_value=commessa_df)
        msg_recived = MagicMock(spec=Event, msg_type='text', msg_body='testo messaggio', sender='111')
        
        self.agent.manageRecivedMessage(msg_recived)

        mock_analyze.assert_called_once()

    @patch('whatsapp_agent.Whatsapp_Agent.send_message')
    def test_manageRecivedMessage_multiple_commesse_no_match(self, mock_send_message):
        """
        Testa il caso di più commesse senza che il messaggio ne specifichi una.
        """
        commessa_df = pd.DataFrame([
            {'commessa': 'C100', 'matricola_tecnico': 'M1', 'cronologia': 1},
            {'commessa': 'C200', 'matricola_tecnico': 'M1', 'cronologia': 1}
        ])
        self.agent.checkLevel = MagicMock(return_value=commessa_df)
        msg_recived = MagicMock(spec=Event, msg_type='text', msg_body='ho lavorato 8 ore', sender='111')
        
        self.agent.manageRecivedMessage(msg_recived)

        mock_send_message.assert_called_once()
        # Verifica che il messaggio di richiesta chiarimenti venga inviato
        sent_message = mock_send_message.call_args.kwargs['message']
        self.assertIn("Non siamo stati ingrado di indetificare la commessa", sent_message)
        self.assertIn("C100", sent_message)
        self.assertIn("C200", sent_message)

    @patch('whatsapp_agent.Whatsapp_Agent.analyzeMessage')
    def test_manageRecivedMessage_multiple_commesse_with_match(self, mock_analyze):
        """
        Testa il caso di più commesse dove il messaggio ne specifica una.
        """
        commessa_df = pd.DataFrame([
            {'commessa': 'C100', 'matricola_tecnico': 'M1', 'cronologia': 1},
            {'commessa': 'C200', 'matricola_tecnico': 'M1', 'cronologia': 1}
        ])
        self.agent.checkLevel = MagicMock(return_value=commessa_df)
        msg_recived = MagicMock(spec=Event, msg_type='text', msg_body='per la commessa C200 ho lavorato 8 ore', sender='111')

        self.agent.manageRecivedMessage(msg_recived)

        mock_analyze.assert_called_once()
        # Verifica che analyzeMessage sia stato chiamato con la commessa corretta
        analyze_args = mock_analyze.call_args.args[0]
        self.assertEqual(analyze_args['commessa'], 'C200')

    @patch('whatsapp_agent.Event')
    def test_manage_message_async_event_failed_first_time(self, MockEvent):
        """
        Test: evento failed con failed_count=0 (prima volta)
        Deve chiamare sendMessageAgain
        """
        # Setup del mock Event
        mock_evento = MagicMock()
        mock_evento.event_type = 'event'
        mock_evento._status = 'failed'
        mock_evento.msg_status = 'failed'
        mock_evento.sender = '391234567890'
        mock_evento.msg_id = 'test_msg_123'
        MockEvent.return_value = mock_evento

        # Setup del mock per message_db.find()
        self.agent.message_db.find.return_value = self._create_commessa_dataframe(failed_count=0)

        # Mock di sendMessageAgain - IMPORTANTE: mock direttamente sul metodo dell'istanza
        self.agent.sendMessageAgain = MagicMock()

        # Dati di input (non importa il contenuto perché Event è mockato)
        data = {"dummy": "data"}

        # Esegui il metodo
        self.agent.manageMessageAsync(data)

        # Debug: stampa cosa è successo
        # Debug più dettagliato
        print(f"logger.info chiamato: {self.agent.logger.info.called}")
        print(f"logger.info args: {self.agent.logger.info.call_args_list}")
        print(f"logger.error chiamato: {self.agent.logger.error.called}")
        print(f"logger.error args: {self.agent.logger.error.call_args_list}")

        # Verifiche
        MockEvent.assert_called_once_with(data)
        self.agent.message_db.find.assert_called_once()
        self.agent.sendMessageAgain.assert_called_once()
        
    @patch('whatsapp_agent.Event')
    @patch('whatsapp_agent.time')
    def test_manage_message_async_event_failed_second_time(self, mock_time, MockEvent):
        """
        Test: evento failed con failed_count=1 (seconda volta)
        Deve attendere 5 minuti e poi chiamare sendMessageAgain
        """
        # Setup del mock Event
        mock_evento = MagicMock()
        mock_evento.event_type = 'event'
        mock_evento._status = 'failed'
        mock_evento.msg_status = 'failed'
        mock_evento.sender = '391234567890'
        mock_evento.msg_id = 'test_msg_123'
        MockEvent.return_value = mock_evento
        
        # Setup del mock per message_db.find() con failed_count=1
        self.agent.message_db.find.return_value = self._create_commessa_dataframe(failed_count=1)
        
        # Mock di sendMessageAgain
        self.agent.sendMessageAgain = MagicMock()

        # Dati di input (non importa il contenuto perché Event è mockato)
        data = {"dummy": "data"}
        
        # Esegui il metodo
        data = self._create_event_data_failed()
        self.agent.manageMessageAsync(data)

        # Debug: stampa cosa è successo
        # Debug più dettagliato
        print(f"logger.info chiamato: {self.agent.logger.info.called}")
        print(f"logger.info args: {self.agent.logger.info.call_args_list}")
        print(f"logger.error chiamato: {self.agent.logger.error.called}")
        print(f"logger.error args: {self.agent.logger.error.call_args_list}")
        
        # Verifiche
        mock_time.sleep.assert_called_once_with(60*5)  # 5 minuti
        self.agent.sendMessageAgain.assert_called_once()
        
    @patch('whatsapp_agent.Event')
    def test_manage_message_async_event_failed_too_many_times(self, MockEvent):
        """
        Test: evento failed con failed_count>=2
        Non deve chiamare sendMessageAgain, solo log warning
        """
        # Setup del mock Event
        mock_evento = MagicMock()
        mock_evento.event_type = 'event'
        mock_evento._status = 'failed'
        mock_evento.msg_status = 'failed'
        mock_evento.sender = '391234567890'
        mock_evento.msg_id = 'test_msg_123'
        MockEvent.return_value = mock_evento
        
        # Setup del mock per message_db.find() con failed_count=2
        self.agent.message_db.find.return_value = self._create_commessa_dataframe(failed_count=2)
        
        # Mock di sendMessageAgain
        self.agent.sendMessageAgain = MagicMock()

        # Dati di input (non importa il contenuto perché Event è mockato)
        data = {"dummy": "data"}

        # Esegui il metodo
        data = self._create_event_data_failed()
        self.agent.manageMessageAsync(data)

        # Debug: stampa cosa è successo
        # Debug più dettagliato
        print(f"logger.info chiamato: {self.agent.logger.info.called}")
        print(f"logger.info args: {self.agent.logger.info.call_args_list}")
        print(f"logger.warning chiamato: {self.agent.logger.warning.called}")
        print(f"logger.warning args: {self.agent.logger.warning.call_args_list}")
        
        # Verifiche
        self.agent.sendMessageAgain.assert_not_called()
        self.agent.logger.warning.assert_called_once()
        
    @patch('whatsapp_agent.Event')
    def test_manage_message_async_event_failed_db_not_found(self, MockEvent):
        """
        Test: evento failed ma messaggio non trovato nel DB
        Deve fare log di errore e ritornare
        """
        # Setup del mock Event
        mock_evento = MagicMock()
        mock_evento.event_type = 'event'
        mock_evento._status = 'failed'
        mock_evento.msg_status = 'failed'
        mock_evento.sender = '391234567890'
        mock_evento.msg_id = 'test_msg_123'
        MockEvent.return_value = mock_evento
        
        # Setup del mock per message_db.find() - ritorna None
        self.agent.message_db.find.return_value = None
        
        # Mock di sendMessageAgain
        self.agent.sendMessageAgain = MagicMock()
        
        # Esegui il metodo
        data = self._create_event_data_failed()
        self.agent.manageMessageAsync(data)

        # Debug: stampa cosa è successo
        # Debug più dettagliato
        print(f"logger.info chiamato: {self.agent.logger.info.called}")
        print(f"logger.info args: {self.agent.logger.info.call_args_list}")
        print(f"logger.error chiamato: {self.agent.logger.error.called}")
        print(f"logger.error args: {self.agent.logger.error.call_args_list}")
        
        # Verifiche
        self.agent.sendMessageAgain.assert_not_called()
        # Verifica che sia stato loggato l'errore
        self.agent.logger.error.assert_called()

        
    @patch('whatsapp_agent.Event')
    def test_manage_message_async_event_not_failed(self, MockEvent):
        """
        Test: evento non failed (es. 'delivered', 'read')
        Deve solo fare update dello status
        """
        # Setup del mock Event
        mock_evento = MagicMock()
        mock_evento.event_type = 'event'
        mock_evento._status = 'delivered'
        mock_evento.msg_status = 'delivered'
        mock_evento.sender = '391234567890'
        mock_evento.msg_id = 'test_msg_123'
        MockEvent.return_value = mock_evento
        
        # Esegui il metodo
        data = self._create_event_data_failed()  # I dati non importano, usiamo il mock
        self.agent.manageMessageAsync(data)
        
        # Verifiche - deve chiamare update per aggiornare lo status
        self.agent.message_db.update.assert_called_once()
        
    def test_manage_message_async_invalid_data(self):
        """
        Test: dati malformati che causano AttributeError
        Deve fare log dell'errore
        """
        # Dati invalidi che causeranno un errore in Event
        invalid_data = {"invalid": "data"}
        
        # Esegui il metodo
        self.agent.manageMessageAsync(invalid_data)

        # Debug: stampa cosa è successo
        # Debug più dettagliato
        print(f"logger.info chiamato: {self.agent.logger.info.called}")
        print(f"logger.info args: {self.agent.logger.info.call_args_list}")
        print(f"logger.error chiamato: {self.agent.logger.error.called}")
        print(f"logger.error args: {self.agent.logger.error.call_args_list}")
        
        # Verifica che sia stato loggato l'errore
        self.agent.logger.error.assert_called()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
