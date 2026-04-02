from datetime import datetime
from zoneinfo import ZoneInfo

TIMEZONE = 'Europe/Rome'
class Event:
    """
    Classe dei messaggi ricevuto da whatsapp
    """
    def __init__(self, data):
        self._data = data
        self.setup()

    def __str__(self):
        """
        Print della classe
        """
        if self.event_type == 'message':
            return "tipo evento: {}\n sender: {}\n tipo messaggio: {}\n id messaggio: {}\n orario invio: {}\n".format(self.event_type, self.sender, self.msg_type, self.msg_id, self.send_time)

    def setup(self):
        """
        setting dei valori
        """
        self.event_type = None
        self._entry = self._data['entry'][0]['changes'][0]['value']
        if  'messages' in self._entry.keys():
            # aggiungiamo il messaggio
            self.event_type = 'message'
            self._message = self._entry['messages'][0]
            self.sender = self._message['from']
            self.msg_type = self._message['type']
            self.msg_id = self._message['id']
            unix_timestamp = int(self._message['timestamp'])
            self.send_time = datetime.fromtimestamp(unix_timestamp, tz=ZoneInfo(TIMEZONE))

            # controllo dei messagi
            if self.msg_type == 'text':
                # messagio di testo
                self.msg_body = self._message['text']['body']
            elif self.msg_type == 'audio':
                # messaggio audio
                self._audio = self._message['audio']
                self.audio_id = self._audio['id']
        elif 'statuses' in self._entry.keys():
            # un evento ( invio del messaggio, ricezione, lettura)
            self.event_type = 'event'
            self._statuses = self._entry['statuses'][0]
            self.msg_id = self._statuses['id']
            self.msg_status = self._statuses['status']
            self.sender = self._statuses['recipient_id']
    
    

    
    
     


