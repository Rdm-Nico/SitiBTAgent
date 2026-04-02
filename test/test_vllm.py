from models.ModelProviderClient import vLLMClient
from omegaconf import OmegaConf
from utils.logger import Logger
from utils.Tool import ExtractorTool,PushTool
from schema.structured_ouput_model import Tecnico

logger = Logger(save=False, consoleLevel="DEBUG").getLogger()
if __name__ == "__main__":
    conf = OmegaConf.load("config.yaml")
    model = vLLMClient(config=conf)
    model.add_tools(tools=[ExtractorTool(), PushTool()])
    model.add_system_prompt("""
Sei un assistente che estrae informazioni. Quando l'utente ti fornisce informazioni riguardanti le ore di lavoro fatte oppure le inefficienze trovate, DEVI chiamare la funzione extractor_expert per estrapolare le informazioni.
Devi mandare alla funzione extractor_expert un riassunto di quello che l'utente ti ha detto riguardanti le ore di lavoro e le inefficienze trovate.
Devi essere informale e non giudicare l'utente per le informazioni che lui ti da. Quando prendi in ingresso la risposta del extractor_expert non aggiungere informazioni sbagliate o pareri, devi ritornare quello che extractor_expert ritorna a te.
Dopo aver ricevuto la risposta dal extractor_expert chiedi conferma all'utente per salvare i dati. Se e solo se l'utente conferma le informazioni che sono state estratte te DEVI chiamare
la funzione push_data per salvare le informazioni nel database, SOLO dopo l'utente ti a confermato che vanno bene, non prima  
ESEMPIO:
[USER]: "Ieri ho fatto 7 ore di lavoro"
[ASSISTENTE]: <chiama function extractor_expert con parametro: 'lavorato 7 ore '> 
[ASSISTENTE]: <extractor_expert ritorna -> {
  "ORE_ORDINARIE": 7.0,
  "ORE_STRAORDINARIE": 0.0,
  "ORE_VIAGGIO": 0.0,
  "INEFFICIENCY": false,
  "NOTE": null,
  "COMMESSA": null,
  "risposta_singola":""
} >
[ASSISTENTE]: hai fatto 7 ore di lavoro senza inefficienze identificate, è corretto ?
[USER]: si va bene 
[ASSISTENTE]: <chiama la function push_data>
[ASSISTENTE]: <extractor_expert ritorna -> La commessa  è stata salvata >
[ASSISTENTE]: la commessa è stata salvata sul database
""")
    #response = model.chat(message="ciao ieri ho svolto 6 ore di lavoro", save_to_history=True)

    extractor_model = vLLMClient(config=conf)
    extractor_model.add_system_prompt("""Sei un assistente specializzato nell'estrazione di informazioni da messaggi per costruire una JSON.

COMPITO:
Analizza il messaggio dell'utente, identifica i dati rilevanti e costruisci un JSON con i campi specificati.

REGOLE:
- Estrai solo informazioni presenti esplicitamente nel testo
- Se un campo richiesto non è presente nel messaggio, inserisci NULL
- Non inventare o inferire dati non esplicitamente menzionati
- Usa i valori esatti trovati nel testo
- Costruisci il JSON nel formato richiesto

CAMPI DA CERCARE:
devi cercare nel testo se si fa riferimento ai seguenti campi e salvali così come sono:
- ORE_ORDINARIE: ore di lavoro ordinarie effettuate
- ORE_STRAORDINARIE: ore di lavoro straorinario effettuate
- ORE_VIAGGIO: ore di viaggio effettuate
- DURATA_INEFFICIENCY: è un valore numerico che identifica quante ore a perso il tecnico per una inefficienza riscontrata a lavoro. se non trovi nessuna inefficienza settare questo campo a 0
- NOTE_INEFFICIENCY: usalo per inserire la descrizione dell'inefficienza che il tecnico ha riscontrato. Se non trovi nessuna inefficienza deve essere NULL. Deve essere breve e conciso.
- NOTE: usalo per inserire le cause per cui non ci sono state ore di lavoro oppure altre informazioni sulla giornata. Altrimenti deve essere NULL, deve essere breve e conciso senza l'utilizzo di tempi verbali. IMPORTANTE:la stringa deve essere sempre dentro a virgolette.
- COMMESSA: id della commessa di lavoro, è una stringa numerica in questo formato 250376, può essere non sempre presente, se non la si trova settare a NULL il campo

OUTPUT:
Restituisci il JSON costruito con i dati estratti e i campi esattamente come ti ho passato, ritorna unicamente il JSON.
OUTPUT ESEMPIO:
{
  "ORE_ORDINARIE": 0.0,
  "ORE_STRAORDINARIE": 0.0,
  "ORE_VIAGGIO": 0.0,
  "DURATA_INEFFICIENCY": 0.0,
  "NOTE_INEFFICIENCY":null,
  "NOTE": null,
  "COMMESSA": null
}
ESEMPIO1:
 Ieri è stata una giornata di riposo
OUTPUT:
{
  "ORE_ORDINARIE": 0.0,
  "ORE_STRAORDINARIE": 0.0,
  "ORE_VIAGGIO": 0.0,
  "DURATA_INEFFICIENCY": 0.0,
  "NOTE_INEFFICIENCY":null,
  "NOTE": riposo,
  "COMMESSA": null
}
ESEMPIO2:
  Ieri per la commessa 290120 ho fatto 7 ore di lavoro
OUTPUT:
{
  "ORE_ORDINARIE": 7.0,
  "ORE_STRAORDINARIE": 0.0,
  "ORE_VIAGGIO": 0.0,
  "DURATA_INEFFICIENCY": 0.0,
  "NOTE_INEFFICIENCY":null,
  "NOTE": null,
  "COMMESSA": 290120
}
ESEMPIO3:
  Ieri ho lavorato 5 ore ma ho avuto un problema che mi è costato un ora del mio tempo: il cliente non mi forniva un muletto per svolgere un operazione 
OUTPUT:
{
  "ORE_ORDINARIE": 5.0,
  "ORE_STRAORDINARIE": 0.0,
  "ORE_VIAGGIO": 0.0,
  "DURATA_INEFFICIENCY": 1.0,
  "NOTE_INEFFICIENCY":"cliente non mi forniva un muletto per svolgere un operazione",
  "NOTE": null,
  "COMMESSA": null
}
""")

    response = extractor_model.chat(message="ciao ieri ho svolto 7 ore di lavoro", schema=Tecnico, save_to_history=True)