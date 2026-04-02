import os
import time
import pandas as pd
from models.ModelProviderClient import clientRouter, InferenceClient
from omegaconf import OmegaConf
import random
import time
from tqdm import tqdm
import jsonlines
import json
import warnings
from utils.logger import Logger
warnings.simplefilter(action='ignore', category=FutureWarning)
logger = Logger(save=False, consoleLevel="INFO").getLogger()
CSV_LABEL_FILENAME='./data/second_test/raw_data_combinated_with_less_inf.csv'
OUTPUT_FILE='./data/training_dataset/train/train_generated.jsonl'
MODEL="gpt-oss:20b"
CURRENT_RUN_FILE = "./data/training_dataset/train/config_run_edge_no_explain_spanish.json"
# scegli la modalità
run_schema = {
    'break_step': 30,
    'max_h_l': 4,
    'vertical_l':5,
    'languages': [{
                    'name': 'italian',
                    'modalites': [('normal', 450),('normal_w_correction', 220),('edge_no_explain', 60)],
                    'frasi': {
                        'confirm_words': ['si', 'va bene', 'è  corretto', 'mi sembra giusto', 'ok','si lo puoi inviare'],
                        'incorrect_words': ['no', 'non va bene', 'hai sbagliato']
                    }
                },{
                    'name': 'english',
                    'modalites': [('normal', 145), ('normal_w_correction', 100),('edge_no_explain', 15)],
                    'frasi': {
                        'confirm_words': ['yes','all right','it is correct','it seems right to me','ok','yes you can send it'],
                        'incorrect_words': ['no','it is not okay','you made a mistake']
                    } 
                    
                },{
                    'name': 'spanish',
                    'modalites': [('normal', 145),('normal_w_correction', 100),('edge_no_explain', 15)],
                    'frasi': {
                    'confirm_words':['sí','está bien','es correcto','me parece correcto','ok','sí lo puedes enviar'],
                    'incorrect_words':['no','no está bien','te equivocaste']
                    } 
    }]
}
conf = OmegaConf.load("config.yaml")
IMPORTANT_COLUMNS = {
    "ORE_ORDINARIE" : "ore_ordinarie",
    "ORE_STRAORDINARIE": "ore_straordinarie",
    "ORE_VIAGGIO": "ore_viaggio",
    "INEFFICIENCY": "durata_inefficienza",
    "HOURS_BY_CLIENT": "note_inefficienza",
    "NOTE": "note"
}

REPLY_NOT_GOOD_SYSTEM_PROMPT="""
   ### ROLE AND OBJECTIVE
You are a field technician messaging your supervisor via a chat app.
You have just sent a sequence of messages (PREVIOUS_CONTEXT), but you realized that the other person make a mistake.
Your task is to generate ONE single, natural follow-up message to correct the error or add the missing information (NEW_DATA).

### INPUT FORMAT
The user will provide three fields:
1. **Language**: The target language for your response (e.g., Italian, English, Spanish).
2. **PREVIOUS_CONTEXT**: A list of messages the other user send.
3. **NEW_DATA**: The data you need to add or correct.

### SETTINGS
**TONE:** Informal, colloquial, hasty.
**FORMAT:** Plain text only. No emojis. Lowercase is acceptable.

### STYLE GUIDELINES (Apply to Target Language)
1. **IMMEDIACY:** The message must sound like it was sent 10 seconds after the previous ones. Use natural "connectors" specific to the defined Language.
2. **NO REPETITION:** Do NOT repeat information already present in the PREVIOUS_CONTEXT. Assume the supervisor has read it. Only mention the change/addition.
3. **NATURAL SPELLING:** You can use abbreviations or slight stylistic imperfections typical of chat (but keep the data accurate).
4. **VARIATION:**
   - Sometimes be very brief ("ah, 5 hours").
   - Sometimes explain slightly ("forgot to say I was on press 5").
   - Sometimes correct yourself ("no wait, it was 2 hours").

### DATA HANDLING RULES
1. **NULL/ZERO:** If a numeric value in NEW_DATA is 0 or null, ignore it completely.
2. **DECIMAL HOURS:** If "ORE" (hours) are decimal (e.g., 0.5), convert them to minutes in the text (e.g., "30 minutes" or "half an hour") in the target language.
3. **TRANSLATION:** If a field text description inside NEW_DATA is in a different language than the target Language, translate the concept naturally into the target Language.

### EXAMPLES

Input:
Language: Italian
PREVIOUS_CONTEXT: ['Ieri ho lavorato 6 ore tutto ok']
NEW_DATA: {ore_ordinarie: 7.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
Output:
hai sbagliato, erano 7 le ore non 6.

Input:
Language: Italian
PREVIOUS_CONTEXT: ['Ieri ho lavorato 5 ore e ho fatto un ora di viaggio']
NEW_DATA: {ore_ordinarie: 0, ore_straordinarie : 0, ore_viaggio : 2.0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
Output:
Ho fatto 2 ore di viaggio non 1

Input:
Language: Italian
PREVIOUS_CONTEXT: ['Ho lavorato quattro ore riparando il mandrino, ho viaggiato per un ora e ho fatto tre ore di staordinari']
NEW_DATA: {ore_ordinarie: 8.0, ore_straordinarie : 1.0, ore_viaggio : 0.5, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
Output:
ah hai sbagliato ho fatto otto ore di lavoro, poi un ora di straordinarie e mezz ora di viaggio

Input:
Language: English
PREVIOUS_CONTEXT: ['Closed the cabinet in the company']
NEW_DATA: {ore_ordinarie: 0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 2.0, note_inefficienza: \"guasto elettrico\", note: \"\" }
Output:
No sorry, you forget to insert also the inefficiency that lasted 2 hours due to an electrical failure

Input:
Language: Italian
PREVIOUS_CONTEXT: ['Ho lavorato per 3 ore e ho  fatto il fissaggio del diffusore']
NEW_DATA: {ore_ordinarie: 5.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
Output:
non ti ho scritto che ci ho messo 3 ore, c è ne ho messe 5
"""
REPLY_GOOD_SYSTEM_PROMPT="""
   ### ROLE AND OBJECTIVE
You are a field technician messaging your supervisor via a chat app.
You have just sent a sequence of data (PREVIOUS_CONTEXT), but you realized you forgot to mention something or made a mistake.
Your task is to generate ONE single, natural follow-up message to correct the error or add the missing information (NEW_DATA).

### INPUT FORMAT
The user will provide three fields:
1. **Language**: The target language for your response (e.g., Italian, English, Spanish).
2. **PREVIOUS_CONTEXT**: The data you already send.
3. **NEW_DATA**: The data you need to add or correct.

### SETTINGS
**TONE:** Informal, colloquial, hasty.
**FORMAT:** Plain text only. No emojis. Lowercase is acceptable.

### STYLE GUIDELINES (Apply to Target Language)
1. **IMMEDIACY:** The message must sound like it was sent 10 seconds after the previous ones. Use natural "connectors" specific to the defined Language (e.g., in Italian: "ah scusa", "dimenticavo", "no aspe"; in English: "oh wait", "my bad").
2. **NO REPETITION:** Do NOT repeat information already present in the PREVIOUS_CONTEXT. Assume the supervisor has read it. Only mention the change/addition.
3. **NATURAL SPELLING:** You can use abbreviations or slight stylistic imperfections typical of chat (but keep the data accurate).
4. **VARIATION:**
   - Sometimes be very brief ("ah, 5 hours").
   - Sometimes explain slightly ("forgot to say I was on press 5").
   - Sometimes correct yourself ("no wait, it was 2 hours").

### DATA HANDLING RULES
1. **NULL/ZERO:** If a numeric value in NEW_DATA is 0 or null, ignore it completely.
2. **DECIMAL HOURS:** If "ore" (hours) are decimal (e.g., 0.5), convert them to minutes in the text (e.g., "30 minutes" or "half an hour") in the target language.
3. **TRANSLATION:** If a field text description inside NEW_DATA is in a different language than the target Language, translate the concept naturally into the target Language.
4. **NATURAL DURATION:** if "ore"(hours) are presents(e.g.,2) sometimes you can convert them in a duration between two times for make it more real (e.g. "I work from 8:00 to 10:00) in the target languge.

### EXAMPLES

Input:
Language: Italian
PREVIOUS_CONTEXT: {ore_ordinarie: 6.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
NEW_DATA: {ore_ordinarie: 7.0, ore_straordinarie : 0, ore_viaggio : 3.0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
Output:
scusa ho sbagliato a scrivere, erano 7 le ore. e poi ci ho messo 3 ore di viaggio

Input:
Language: Italian
PREVIOUS_CONTEXT: {ore_ordinarie: 5.0, ore_straordinarie : 0, ore_viaggio : 1.0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" }
NEW_DATA: {ore_ordinarie: 5.0, ore_straordinarie : 0, ore_viaggio : 1.0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"pressa 6 aggiungere l'olio\" }
Output:
dimenticavo la nota: ero sulla pressa 6 per aggiungere l'olio

Input:
Language: Italian
PREVIOUS_CONTEXT: {ore_ordinarie: 4.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"'Ho riparato il mandrino\" }
NEW_DATA: {ore_ordinarie: 4.0, ore_straordinarie : 0, ore_viaggio : 0.5, durata_inefficienza: 0, note_inefficienza: \"Ho riparato il mandrino\", note: \"\" }
Output:
ah e ho fatto anche mezz'oretta di viaggio al ritorno

Input:
Language: English
PREVIOUS_CONTEXT: {ore_ordinarie: 8.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\" } 
NEW_DATA: {ore_ordinarie: 8.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 2.0, note_inefficienza: \"guasto elettronico\", note: \"\" }
Output:
No sorry, there was an inefficiency that lasted 2 hours due to an electrical failure

Input:
Language: Italian
PREVIOUS_CONTEXT: {ore_ordinarie: 0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"Ho fatto il fissaggio del diffusore\" } 
NEW_DATA: {ore_ordinarie: 5.0, ore_straordinarie : 0, ore_viaggio : 0, durata_inefficienza: 0, note_inefficienza: \"\", note: \"Ho fatto il fissaggio del diffusore\" }
Output:
non ti ho scritto quanto ci ho messo: 5 ore totali
"""
GENERATE_MSG_SYSTEM_PROMPT="""
### ROLE
You are a simulator of a field technician sending a text message to their company to report daily activities. You must generate realistic, human-like messages based on the data provided.

### OBJECTIVE
Create a SINGLE message simulating a technician reporting their work. The style must vary per response (formal, informal, hasty, complaining, or detailed).

### INPUT DATA PROCESSING RULES
1.  **Zero/Null Values:** If a numeric value is 0 or a text field is null/empty, DO NOT mention it in the message.
2.  **Time Conversion:** - Never write decimal hours (e.g., "1.5 hours"). 
    - Convert decimal parts to minutes (e.g., 0.5 -> "30 minutes", 0.75 -> "45 minutes").
    - Use natural phrasing (e.g., "5.5" becomes "5 hours and a half" or "5 hours 30").
3.  **Inefficiencies:** If `durata_inefficienza` > 0:
    - You MUST clearly highlight the problem.
    - Use phrases like "I had a problem," "waste of time," "setback," or "bad delay."
    - Include the description of the inefficiency naturally.
4.  **Language Translation:** If the input data description is in a different language than the target output language, translate the concept naturally into the target language.

### STYLE & TONE GUIDELINES
- **Natural Language:** Use colloquialisms appropriate for a worker.
- **NO Emojis:** Strictly forbidden.
- **Variation (Pick one per message):**
    - *Style A (Hasty):* Short, lowercase, lack of punctuation, potential typos.
    - *Style B (Formal):* Polite, clear, complete sentences.
    - *Style C (Annoyed):* Focuses heavily on the inefficiency/problems.
    - *Style D (Chatty):* Adds unnecessary context or small talk.
- **Target Language:** Generate the message in the language specified by the user. If not specified, DEFAULT to **Italian**.

### OUTPUT FORMAT
- Return ONLY the simulated message content.
- Do not add introductions, explanations, or quotes around the output.

### EXAMPLES (Few-Shot)

**User Input:** Language: Italian
Data: {ore_ordinarie: 8, ore_straordinarie: 0, ore_viaggio: 0, durata_inefficienza: 1.0, note_inefficienza: "cliente non mi permetteva di usare il muletto", note: \"\"}
**Assistant Output:**
ieri ho fatto le mie otto ore ma ho perso un'ora piena perchè il cliente non mi sganciava il muletto.

**User Input:** Language: Italian
Data: {ore_ordinarie: 5.5, ore_straordinarie: 0, ore_viaggio: 0.75, durata_inefficienza: 0, note_inefficienza: \"\", note: \"\"}
**Assistant Output:**
Lavoro completato in 5 ore e mezza. Ho anche 45 minuti di viaggio. Tutto liscio.

**User Input:** Language: English
Data: {ore_ordinarie: 4, ore_straordinarie: 2, ore_viaggio: 0, durata_inefficienza: 0.5, note_inefficienza: "no electricity", note: "finished early"}
**Assistant Output:**
did 4 hours regular and 2 overtime. lost 30 mins because there was no power.. managed to finish early tho
"""
GENERATE_SUMMARY_SYSTEM_PROMPT="""
### ROLE
You are a summarization agent of a field technician and you generate accurate summary of the text that you recive in input. You must be the more accurate as possible.

### OBJECTIVE
Create a SINGLE message  that summary the list of messages that you recive. The style must vary per response formal and concise.

### INPUT DATA PROCESSING RULES
1.  **Duplicated content:** If a part of the messages is common, DONT repeat yourself, just put the information one time only.
2.  **Time Conversion:** - If there is some hours always convert in decimal hours (e.g., "1.5 hours"). 
    - Convert decimal parts to minutes (e.g., "30 minutes" ->  0.5, "45 minutes" -> 0.75).
    - if the duration is mark like the time pass between two times , just convert it to decimal (e.g., "I work from 8:00 to 14:00" -> 6 hours, "I work from 14:30 to 17:00" -> 3.5 hours )
3.  **Inefficiencies:** If you find a reference to a inefficiency or anything that have a negative contribution:
    - You MUST clearly highlight the problem.
    - And summary in the most similar way respect to the original part
4.  **Language Translation:** If the input data description is in a different language than the target output language, translate the concept naturally into the target language.

### STYLE & TONE GUIDELINES
- **Natural Language:** Use colloquialisms appropriate for a summarization task.
- **NO Emojis:** Strictly forbidden.
- **Target Language:** Generate the message in the language specified by the user. If not specified, DEFAULT to **Italian**.

### OUTPUT FORMAT
- Return ONLY the simulated message content.
- Do not add introductions, explanations, or quotes around the output.

### EXAMPLES (Few-Shot)

**User Input:** Language: Italian
Data: ['ieri ho fatto 4 ore straordinarie. Ho atteso che il frantoi sia riparato, ho fatto i test ultimi sul mulino 3, carico del mulino N\u00b02 e pianificato il start\u2011up per luned\u00ec mattina, ma il carico del mulino N\u00b03 non \u00e8 finito a causa del blackout']
**Assistant Output:**
fatto otto ore di lavoro più quattro ore di straordinario. atteso che il frantoi sia riparato, fatto i test ultimi sul mulino 3, carico del mulino N\u00b02 e pianificato il start\u2011up per luned\u00ec mattina, ma il carico del mulino N\u00b03 non \u00e8 finito a causa del blackout.

**User Input:** Language: Italian
Data: ['Ieri ho fatto 8 ore ordinarie con prensenza in cantiere', 'Aggiungi anche due ore di straordinari']
**Assistant Output:**
fatto otto ore più due ore di straordinari con presenza in cantiere

**User Input:** Language: English
Data: ['I worked from 8:00 to 12:00 at the gas station. I had a problem that last one hour regarding the water pomp']
**Assistant Output:**
did 4 hours at the gas station. Had a inefficiency of 1 hour regarding the water pomp
"""

"""
File per creare messaggi artificiali basandosi sui dati utili del file CSV_LABEL_FILENAME
"""

def upload_label():
    """Upload delle label del csv"""
    data = pd.read_csv(CSV_LABEL_FILENAME)
    # aggiungiamo l'indice di riferminento che sarà poi utilizzato dopo:
    data['REF_INDICE'] = None
    logger.debug(f'data returned {data.shape}. Columns: {data.columns}')
    return data


def restart_from(index):
    """
    Scarica i dati da un punto specifico del db
    """
    data = pd.read_csv(CSV_LABEL_FILENAME)

    pos = data.loc[ data['INDICI'] == index ].index[0]

    return data.loc[pos:]
    

def generate_msg(row:pd.Series, model_first:InferenceClient, model_reply:InferenceClient, model_summary:InferenceClient, masks_weights:dict, modality:str, mods:dict, language:str, frasi:dict) -> dict:
    """generazione del flusso di messaggi da parte di gpt.
       - Generare messaggi a seconda del flusso dei messaggi (verticale) 
       - A seconda di quanti campi cambiare in ogni messaggio (orrizzontale)
       - Generare Edge Case 
            - caso in cui si dice al modello che sbaglia senza specificare nulla
            - caso in cui si dice al modello che sbaglia e si specifica cosa cambiare 
            - caso in cui si dice una cosa che non centra niente al modello 

    """
    output_dict = {}
    messaggi = []
    

    match modality:
        case 'normal' | 'normal_w_correction':
            """
            creiamo i messaggi artificiali come lista di VERTICAL_L+1  messaggi. 
            La prima riga sarà il messaggio inviato al primo modello (con le maschere eventuali)
            Le altre  VERTICAL_L -1 righe saranno le modifiche fatte al messaggio originale.
            """
            # decrementiamo di uno per tenere conto del messaggio finale 
            flow_len = run_schema['vertical_l'] -1
            modified_rows = pd.DataFrame(columns=row.index)
            changed_fileds = []
            not_show_fields = list(masks_weights.keys())
            for i in range(flow_len):
                """ per ogni riga del messaggio artificiale dobbiamo:
                    - (selezionare l'elemento da modificare dalla riga originale)x horziontal_l
                    - dobbiamo verificare che len(masks_weights) sia >= horziontal_l per non incappare in loop infiniti
                    - (aggiungere alla riga del df di msg artificiali  il campo modificato)x horziontal_l
                    - una lista already_used_fields contiene quali fields sono stati modificati in una step
                    - una lista not_show_fields contiene i campi non ancora mostrati 
                    - non appena un campo originale viene messo nella riga bisogna fare il remove di quel elemento da changed_fields
                    - non appena si inserisce un elemento nella riga si fa il remove di quel elemento da not_show_fields
                    - changed_fileds e not_show_fields deve essere sempre vuoto prima di terminare il primo for loop

                """ 
                already_used_fields = []

                # se abbiamo vuota la lista not_show_fields e masks_weights allora possiamo uscire
                if len(not_show_fields) == 0 and len(masks_weights) == 0:
                    break
                if i == 0:
                    # se siamo al primo messaggio creiamo la riga vuota. Aggiungiamo object per poter accettare tutti i tipi possibili in ingresso
                    mod_row = pd.Series(index=row.index, dtype=object)
                else:
                    # se no creiamo una riga con gli stessi valori di quella precedente così da avere l ultima riga del df con tutti i valori corretti
                    mod_row = modified_rows.iloc[-1].copy()
                   

                # se siamo all'ultima riga  
                if i == flow_len -1:
                    logger.debug(f"siamo all'ultima riga che è la riga {i} con queste liste: changed_fileds:{changed_fileds}\tnot_show_fields:{not_show_fields}")
                    # dobbiamo fare in modo di avere vuoto changed_fileds e not show_fields 
                    while((len(changed_fileds) != 0) or (len(not_show_fields) != 0)):
                        # prendiamo un elemento e lo inseriamo
                        field = None
                        if len(not_show_fields) != 0:
                            idx = random.randrange(0,len(not_show_fields)) 
                            field = not_show_fields.pop(idx)
                        else:
                            idx = random.randrange(0,len(changed_fileds)) 
                            field = changed_fileds.pop(idx)
                        #print(f'field: {field}')    
                        mod_row.at[field] = row.at[field]
                    
                else:
                    horziontal_l = random.randrange(1,run_schema['max_h_l'])
                    for j in range(horziontal_l):
                        logger.debug(f'elementi da modificare nella riga {i} sono {j}/{horziontal_l}. not_show_fields:{not_show_fields}\tchanged_fileds:{changed_fileds}\talready_used_fields: {already_used_fields}\tmasks_weights:{masks_weights}')
                        # se abbiamo finito gli elementi da aggiungere ci stoppiamo
                        if not not_show_fields and not changed_fileds and not already_used_fields:
                            break
                        if len(masks_weights) == 1:
                            # se è già presente nel campo del already_used_fields allora stoppiamo 
                            field_2_mask = random.choices(list(masks_weights.keys()), masks_weights.values())[0]
                            if field_2_mask in  already_used_fields:
                                break
                            # se invece è lo stesso del campo changed_fileds e changed_fileds è uguale a 1
                            if field_2_mask in changed_fileds and len(changed_fileds) == 1:
                                # inseriamo il valore reale
                                mod_row.at[field_2_mask] = row.at[field_2_mask]
                                if field_2_mask in changed_fileds:
                                    changed_fileds.remove(field_2_mask)
                                masks_weights.pop(field_2_mask)
                                break

                                 
                        if not masks_weights:
                            break
                        # selezionare un elemento da modificare o da aggiungere
                        while(1):
                            field_2_mask = random.choices(list(masks_weights.keys()), masks_weights.values())[0]
                            if field_2_mask not in already_used_fields or field_2_mask in not_show_fields or field_2_mask in changed_fileds:

                                # se questo elemento non si trova in not_show_fields e non è in changed_fileds vuole dire che è già stato inserito il valore corretto
                                if field_2_mask not in not_show_fields and field_2_mask not in changed_fileds:
                                    masks_weights.pop(field_2_mask)
                                    continue
                                break


                        # se non siamo all'ultima riga 
                        if i < flow_len -1:
                            # possiamo decidere se modificare o aggiungere il valore del campo 
                            ris = random.choices(list(mods.keys()), mods.values())[0] if field_2_mask != 'NOTE' else 'show'
                            #ris = random.choice(mods) if field_2_mask != 'NOTE' else 'mask'
                            if ris == 'mod':
                                # modifichiamo e inseriamo il campo dentro a changed_fileds se non è presente e already_used_fields e rimuoviamolo da not_show_fields se  è presente
                                dist = random.randrange(-3,+3)
                                dist = 1 if dist == 0 else dist
                                mod_row.at[field_2_mask] = row.at[field_2_mask] +  dist
                                # se è negativo lo mettiamo  a +1 del campo originale
                                mod_row.at[field_2_mask] = row[field_2_mask]+1 if mod_row.at[field_2_mask] < 0 else mod_row.at[field_2_mask]
                                # aggiungiamo alla lista
                                if field_2_mask not in changed_fileds:
                                    changed_fileds.append(field_2_mask)
                                if field_2_mask in not_show_fields:
                                    not_show_fields.remove(field_2_mask)
                            else:
                                # aggiungiamo il campo originale in questo caso e rimuoviamo da changed_fileds se è presente e anche da not_show_fields e masks_weights
                                mod_row.at[field_2_mask] = row.at[field_2_mask]
                                if field_2_mask in changed_fileds:
                                    changed_fileds.remove(field_2_mask)
                                if field_2_mask in not_show_fields:
                                    not_show_fields.remove(field_2_mask)
                                masks_weights.pop(field_2_mask)
                            # aggiungiamo a already_used_fields
                            already_used_fields.append(field_2_mask)
                    
                #print(f"aggiungiamo la riga-{i}: {mod_row}")
                # aggiungiamo la riga al df solo se non è tutta vuota 
                if len(mod_row.loc[mod_row.isna() != True]) != 0:
                    modified_rows.loc[len(modified_rows)] = mod_row
                    
            # rimuoviamo l'ultima riga che è una copia dell'ultima
            logger.debug(f"righe msg artificiali:\n{modified_rows[['ORE_ORDINARIE', 'ORE_STRAORDINARIE', 'ORE_VIAGGIO' , 'NOTE', 'INEFFICIENCY', 'HOURS_BY_CLIENT']]}")
            logger.debug(f"riga originale: {row[['ORE_ORDINARIE', 'ORE_STRAORDINARIE', 'ORE_VIAGGIO' , 'NOTE', 'INEFFICIENCY','HOURS_BY_CLIENT']]}")
            
            previous_found = False
            # init il contesto dell'utente per faciliare il lavoro dell'AI 
            user_context =  []
            prev_data =  {}
            for i in range(len(modified_rows)):
                if i == 0:
                    dict_2_pass = {}
                    # rimuoviamo gli elementi che non ci interessano
                    important_values = modified_rows.iloc[i][IMPORTANT_COLUMNS.keys()]
                    # rimuoviamo gli elementi nulli
                    #fill_values = important_values.loc[important_values.isna() == False]
                    # sostiuiamo gli elementi nulli con spazi oppure zeri
                    fill_values = important_values.fillna(value={"ORE_ORDINARIE": 0, "ORE_STRAORDINARIE": 0, "ORE_VIAGGIO": 0, "INEFFICIENCY": 0, "NOTE": "","HOURS_BY_CLIENT": ""})
                    # facciamo lo zip per generare il dict
                    for column_name,value in zip(fill_values.index, fill_values.to_numpy()):
                        if column_name == 'HOURS_BY_CLIENT':
                            # skippiamo che la gestiamo nel inefficiency
                            continue
                        if column_name == 'INEFFICIENCY' and value > 0:
                            # controlliamo se abbiamo già incontrato l'inefficienza possiamo non includere la descrizione
                            if previous_found:
                                dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                            else:
                                previous_found = True
                                dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                                dict_2_pass[IMPORTANT_COLUMNS['HOURS_BY_CLIENT']] = row['HOURS_BY_CLIENT']
                        else:
                            # nel mentre convertiamo anche il nome della colonna nel formato del structured output
                            dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value

                    # prompt del primo modello
                    prompt = f"Language: {language}\nData:{dict_2_pass}"
                    
                    logger.debug(f"prompt del primo messaggio da creare: {prompt}")
                    response = model_first.chat(message=prompt, verbose=True)
                    logger.debug(f"risposta prima modello: {response['message']['content']}")
                    
                    # aggiungiamo i dati precedenti al nuovo contesto
                    prev_data = dict_2_pass
                    # aggiungiamo il messaggio dell'utente
                    messaggi.append({ "role": "user", "content": response['message']['content']})
                    # aggiungiamo al contesto
                    user_context.append(response['message']['content'])
                    
                    # aggiungiamo la chiamata al tool dell'assistente 
                    prompt_assistant = f"Language: {language}\n Data: {user_context}"
                    logger.debug(f"prompt per creare l argomento da passare al modello che fa il summary: {prompt_assistant}")
                    response = model_summary.chat(message=prompt_assistant, verbose=True)
                    logger.debug(f"response del modello che crea il summary: {response['message']['content']}")
                    # aggiungiamo il messaggio generato dall'assistente
                    assistant_tool_call = {'name': 'extractor_expert', 'arguments': {'summary': response['message']['content']}}
                    assistant_tool_call_str = f"<TOOLCALL>[{json.dumps(assistant_tool_call, ensure_ascii=True)}]</TOOLCALL>" 

                    messaggi.append({"role": "assistant", "content": assistant_tool_call_str})

                    # aggiungiamo il risultato dell'extractor 
                    # aggiungiamo la commessa in tanto
                    dict_2_pass['commessa'] = ""
                    extractor_tool_result = {'name': 'extractor_api', 'arguments': dict_2_pass}

                    extractor_tool_result_str = f"<TOOLCALL>[{json.dumps(extractor_tool_result, ensure_ascii=True)}]</TOOLCALL>" 

                    messaggi.append({"role": "extractor", "content": extractor_tool_result_str})
                    if len(modified_rows) == 1:
                        # allora dobbiamo incluedere anche il messaggio di conferma
                        messaggi.append({ "role": "user", "content": random.choice(frasi['confirm_words'])})
                        # aggiungiamo la chiamata al tool
                        assistant_tool_call = {'name': 'push_data', 'arguments': {'push': True}}
                        assistant_tool_call_str = f"<TOOLCALL>[{json.dumps(assistant_tool_call, ensure_ascii=True)}]</TOOLCALL>" 
                        messaggi.append({"role": "assistant", "content": assistant_tool_call_str})
                        

                else:
                    # rimuoviamo gli elementi che non ci interessano
                    important_values = modified_rows.iloc[i][IMPORTANT_COLUMNS.keys()]
                    dict_2_pass = {}
                    # sostiuiamo gli elementi nulli con spazi oppure zeri
                    fill_values = important_values.fillna(value={"ORE_ORDINARIE": 0, "ORE_STRAORDINARIE": 0, "ORE_VIAGGIO": 0, "INEFFICIENCY": 0, "NOTE": "","HOURS_BY_CLIENT": ""})
                    # facciamo lo zip per generare il dict
                    for column_name,value in zip(fill_values.index, fill_values.to_numpy()):
                        if column_name == 'HOURS_BY_CLIENT':
                            # skippiamo che la gestiamo nel inefficiency
                            continue
                        if column_name == 'INEFFICIENCY' and value > 0:
                            # controlliamo se abbiamo già incontrato l'inefficienza possiamo non includere la descrizione
                            if previous_found:
                                dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                            else:
                                previous_found = True
                                dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                                dict_2_pass[IMPORTANT_COLUMNS['HOURS_BY_CLIENT']] = row['HOURS_BY_CLIENT']
                        else:
                            # nel mentre convertiamo anche il nome della colonna nel formato del structured output
                            dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                    
                 
                    
                    # prompt del secondo modello
                    prompt = f"Language: {language}\nPREVIOUS_CONTEXT: {prev_data}\nNEW_DATA: {dict_2_pass}"
                    logger.debug(f"prompt del {i}esimo messaggio da inviare: {prompt}")

                    response = model_reply.chat(prompt, verbose=True)
                    #print(f"risposta secondo modello: {response['response']}")
                    # aggiungiamo nuovamente alla nuova riga
                    prev_data = dict_2_pass
                    # aggiungiamo il messaggio dell'utente
                    messaggi.append({ "role": "user", "content": response['message']['content']})
                    # aggiungiamo al contesto
                    user_context.append(response['message']['content'])
                    
                    # aggiungiamo la chiamata al tool dell'assistente 
                    prompt_assistant = f"Language: {language}\n Data: {user_context}"
                    logger.debug(f"prompt per creare l argomento da passare al modello che fa il summary: {prompt_assistant}")
                    response = model_summary.chat(message=prompt_assistant, verbose=True)
                    logger.debug(f"response del modello che crea il summary: {response['message']['content']}")
                    # aggiungiamo il messaggio generato dall'assistente
                    assistant_tool_call = {'name': 'extractor_expert', 'arguments': {'summary': response['message']['content']}}
                    assistant_tool_call_str = f"<TOOLCALL>[{json.dumps(assistant_tool_call, ensure_ascii=True)}]</TOOLCALL>" 

                    messaggi.append({"role": "assistant", "content": assistant_tool_call_str})

                    # aggiungiamo il risultato dell'extractor 
                    # aggiungiamo la commessa in tanto
                    dict_2_pass['commessa'] = ""
                    extractor_tool_result = {'name': 'extractor_api', 'arguments': dict_2_pass}

                    extractor_tool_result_str = f"<TOOLCALL>[{json.dumps(extractor_tool_result, ensure_ascii=True)}]</TOOLCALL>" 

                    messaggi.append({"role": "extractor", "content": extractor_tool_result_str})
                    
                    if i == len(modified_rows) -1 :
                        # inseriamo anche il messaggio di conferma
                        messaggi.append({ "role": "user", "content": random.choice(frasi['confirm_words'])})
                        # aggiungiamo la chiamata al tool
                        assistant_tool_call = {'name': 'push_data', 'arguments': {'push': True}}
                        assistant_tool_call_str = f"<TOOLCALL>[{json.dumps(assistant_tool_call, ensure_ascii=True)}]</TOOLCALL>" 
                        messaggi.append({"role": "assistant", "content": assistant_tool_call_str})

            logger.debug(f'riga finale dei messaggi: {messaggi}')

            # aggiungiamo alle righe le info
            output_dict['language'] = language
            output_dict['type'] = modality
            output_dict['messages'] = messaggi 

        
        case 'edge_why':
            """
            Edge case con (msg_init) sbagliato + response dell'utente negativa con spiegazione + confirm msg.

            Passiamo tutti i campi nel primo messaggio con almeno un campo sbagliato
            Poi utilizziamo il secondo modello per correggere la risposta 
            Infine confirm msg
            """
            modified_rows = pd.DataFrame(columns=row.index)
            changed_fields = []
            mod_row = row.copy()
            # cambiamo almeno un campo
            horziontal_l = random.randrange(1,run_schema['max_h_l'])
            # facciamo un loop while in questo modo saimo certi che almeno un campo verrà cambiato
            while(len(changed_fields) == 0):
                weights = masks_weights.copy()
                for j in range(horziontal_l):
                    #print(f"modifichiamo {j}/{MAX_HORIZONTAL_L}\tchanged_fields: {changed_fields}\tweights: {weights}")
                    

                    # se il weights è vuoto usciamo 
                    if len(weights) == 0:
                        break
                    # se abbamo solo due elementi e uno di questi è il campo NOTE, cambiamo direttamente l'altro elemento
                    if len(weights) == 2 and 'NOTE' in weights:
                        weights.pop('NOTE')
                        field_2_mask = random.choices(list(weights.keys()), weights.values())[0]
                        # modifichiamo e inseriamo il campo dentro a changed_fields 
                        dist = random.randrange(-3,+3)
                        dist = 1 if dist == 0 else dist
                        mod_row.at[field_2_mask] = row.at[field_2_mask] +  dist
                        # se è negativo lo incrementiamo di uno rispetto al lavore originale
                        mod_row.at[field_2_mask] = row[field_2_mask]+1 if mod_row.at[field_2_mask] < 0 else mod_row.at[field_2_mask]
                        # aggiungiamo alla lista
                        if field_2_mask not in changed_fields:
                            changed_fields.append(field_2_mask)
                            weights.pop(field_2_mask)
                        break


                    field_2_mask = random.choices(list(weights.keys()), weights.values())[0]
                    #print(f"field_2_mask: {field_2_mask}")
                    
                    # possiamo decidere se modificare  il valore 
                    ris = random.choices(list(mods.keys()), mods.values())[0] if field_2_mask != 'NOTE' else 'show'
                    
                    if ris == 'mod':
                        # modifichiamo e inseriamo il campo dentro a changed_fields 
                        dist = random.randrange(-3,+3)
                        dist = 1 if dist == 0 else dist
                        mod_row.at[field_2_mask] = row.at[field_2_mask] +  dist
                        # se è negativo lo mettiamo a +1
                        mod_row.at[field_2_mask] = row[field_2_mask]+1 if mod_row.at[field_2_mask] < 0 else mod_row.at[field_2_mask]
                        # aggiungiamo alla lista
                        if field_2_mask not in changed_fields:
                            changed_fields.append(field_2_mask)
                            weights.pop(field_2_mask)
                    else:
                        # rimuoviamo da weights
                        weights.pop(field_2_mask)

            # facciamo l'append nel df
            modified_rows.loc[len(modified_rows)] = mod_row

            # inseriamo nella seconda riga tutti i valori mancati corretti
            update_row = pd.Series(index=row.index)
            for field in changed_fields:
                update_row.at[field] = row.at[field]
            # appendiamo 
            modified_rows.loc[len(modified_rows)] = update_row

            #print(f"righe msg artificiali:\n{modified_rows[['ORE_ORDINARIE', 'ORE_STRAORDINARIE', 'ORE_VIAGGIO' , 'NOTE', 'INEFFICIENCY']]}")
            #print(f"riga originale: {row[['ORE_ORDINARIE', 'ORE_STRAORDINARIE', 'ORE_VIAGGIO' , 'NOTE', 'INEFFICIENCY']]}")

            previous_found= False
            for i in range(len(modified_rows)):
                if i == 0:
                    # prompt del primo modello
                    dict_2_pass = {}
                    # rimuoviamo gli elementi che non ci interessano
                    important_values = modified_rows.iloc[i][IMPORTANT_COLUMNS]
                    # rimuoviamo gli elementi nulli
                    fill_values = important_values.loc[important_values.isna() == False]
                    # facciamo lo zip 
                    for column_name,value in zip(fill_values.index, fill_values.to_numpy()):
                        if column_name == 'INEFFICIENCY':
                            # controlliamo se abbiamo già incontrato l'inefficienza possiamo non includere la descrizione
                            if previous_found:
                                dict_2_pass[column_name] = value
                            else:
                                previous_found = True
                                dict_2_pass[column_name] = value
                                dict_2_pass['INEFFICIENCY_DESC'] = row['HOURS_BY_CLIENT']
                        else:
                            dict_2_pass[column_name] = value

                    # prompt del primo modello
                    prompt = f"Language: {language}\nData:{dict_2_pass}"

                    #print(f"prompt primo modello: {prompt}")
                    response = model_first.generate(prompt)
                    
                    
                    # aggiungiamo il messaggio
                    messaggi.append(response['response'])
                else:
                    dict_2_pass = {}
                    # rimuoviamo gli elementi che non ci interessano
                    important_values = modified_rows.iloc[i][IMPORTANT_COLUMNS]
                    # rimuoviamo gli elementi nulli
                    fill_values = important_values.loc[important_values.isna() == False]
                    # facciamo lo zip 
                    for column_name,value in zip(fill_values.index, fill_values.to_numpy()):
                        dict_2_pass[column_name] = value
                    
                    # prompt del secondo modello
                    prompt = f"Language: {language}\nPREVIOUS_CONTEXT: {messaggi}\nNEW_DATA: {dict_2_pass}"

                    #print(f"prompt secondo modello {i} parte: {prompt}")
                    response = model_reply.generate(prompt)
                    #print(f"risposta secondo modello: {response['response']}")
                    messaggi.append(response['response'])
                    if i == len(modified_rows) -1 :
                        # inseriamo anche il messaggio di conferma
                        messaggi.append(random.choice(frasi['confirm_words']))

            #print(f'riga finale dei messaggi: {messaggi}')
            # aggiungiamo alle righe le info
            row['MESSAGGI'] = messaggi
            row['L_MESSAGGI'] = len(messaggi)
            row['W_MESSAGGI'] = len(changed_fields)
            row['TYPE_EVAL'] = modality
            row['LANGUAGE'] = language
            row['REF_INDICE'] = row['INDICE']


        case 'edge_no_explain':
            """Edge case con (msg_init) giusto + response dell'utente negativa senza spiegazione + chiamata al tool false"""
            # prompt del primo modello
            dict_2_pass = {}
            # rimuoviamo gli elementi che non ci interessano
            modified_row = row.copy()
            important_values = modified_row[IMPORTANT_COLUMNS.keys()]
            # rimuoviamo gli elementi nulli
            fill_values = important_values.loc[important_values.isna() == False]
            previous_found= False
            # facciamo lo zip per generare il dict
            for column_name,value in zip(fill_values.index, fill_values.to_numpy()):
                if column_name == 'HOURS_BY_CLIENT':
                    # skippiamo che la gestiamo nel inefficiency
                    continue
                if column_name == 'INEFFICIENCY' and value > 0:
                    # controlliamo se abbiamo già incontrato l'inefficienza possiamo non includere la descrizione
                    if previous_found:
                        dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                    else:
                        previous_found = True
                        dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value
                        dict_2_pass[IMPORTANT_COLUMNS['HOURS_BY_CLIENT']] = row['HOURS_BY_CLIENT']
                else:
                    # nel mentre convertiamo anche il nome della colonna nel formato del structured output
                    dict_2_pass[IMPORTANT_COLUMNS[column_name]] = value

            # prompt del primo modello
            prompt = f"Language: {language}\nData:{dict_2_pass}"
            
            logger.debug(f"prompt del primo messaggio da creare: {prompt}")
            response = model_first.chat(message=prompt, verbose=True)
            logger.debug(f"risposta prima modello: {response['message']['content']}")
            
            # aggiungiamo il messaggio dell'utente
            messaggi.append({ "role": "user", "content": response['message']['content']})

            
            # aggiungiamo la chiamata al tool dell'assistente 
            prompt_assistant = f"Language: {language}\n Data: {response['message']['content']}"
            logger.debug(f"prompt per creare l argomento da passare al modello che fa il summary: {prompt_assistant}")
            response = model_summary.chat(message=prompt_assistant, verbose=True)
            logger.debug(f"response del modello che crea il summary: {response['message']['content']}")
            # aggiungiamo il messaggio generato dall'assistente
            assistant_tool_call = {'name': 'extractor_expert', 'arguments': {'summary': response['message']['content']}}
            assistant_tool_call_str = f"<TOOLCALL>[{json.dumps(assistant_tool_call, ensure_ascii=True)}]</TOOLCALL>" 
            messaggi.append({"role": "assistant", "content": assistant_tool_call_str})
            # aggiungiamo il risultato dell'extractor 
            # aggiungiamo la commessa in tanto
            dict_2_pass['commessa'] = ""
            extractor_tool_result = {'name': 'extractor_api', 'arguments': dict_2_pass}
            extractor_tool_result_str = f"<TOOLCALL>[{json.dumps(extractor_tool_result, ensure_ascii=True)}]</TOOLCALL>" 
            messaggi.append({"role": "extractor", "content": extractor_tool_result_str})


            # aggiungiamo la response negativa senza spiegazioni
            messaggi.append({ "role": "user", "content": random.choice(frasi['incorrect_words'])})
            # passiamo la chiamata al tool con valore uguale a false
            assistant_tool_call = {'name': 'push_data', 'arguments': {'push': False}}
            assistant_tool_call_str = f"<TOOLCALL>[{json.dumps(assistant_tool_call, ensure_ascii=True)}]</TOOLCALL>" 
            messaggi.append({"role": "assistant", "content": assistant_tool_call_str})
            
            logger.debug(f'riga finale dei messaggi: {messaggi}')

            # aggiungiamo alle righe le info
            output_dict['language'] = language
            output_dict['type'] = modality
            output_dict['messages'] = messaggi 
        
        
    return output_dict

def save_current_run_info(curr_idx, language, modality, selected_rows):
    output_run = {
        "current_idx": curr_idx,
        "language": language,
        "modality": modality,
        "selected_rows": selected_rows
    }
    with open(CURRENT_RUN_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_run, f, ensure_ascii=False)
    
    logger.info(f"la corsa attuale salvata in {CURRENT_RUN_FILE} ")


def Preprocessing():
    """Preparazione  del file labels con l'aggiunta di una nuova colonne """
    labels = None
    selected_rows = []
    labels = upload_label()
    # importa modelli 
    client_first_msg = clientRouter(conf, name="generate_training_model")
    client_first_msg.add_system_prompt(GENERATE_MSG_SYSTEM_PROMPT)
    
    client_reply_msg = clientRouter(conf, name="generate_training_model")
    client_reply_msg.add_options({'temperature': 0.7})

    client_summary = clientRouter(conf, name="generate_training_model")
    client_summary.add_system_prompt(GENERATE_SUMMARY_SYSTEM_PROMPT)

    info = None
    if os.path.exists(CURRENT_RUN_FILE):
        # apriamo file della run attuale
        with open(CURRENT_RUN_FILE) as f:
            info = json.load(f)
            logger.info(f"checkpoint file caricato correttamente con valori: {info}")
        
        

    
    start_time = time.time()
    for language in run_schema['languages']:
        logger.info('-' * 100)
        logger.info(f"prepariamo il dataset per la linuga {language['name']}")
        logger.info('-' * 100)


        for modality, row_2_create in language['modalites']:
            logger.info('+' * 100)
            logger.info(f"modalità: {modality} righe da creare: {row_2_create}")
            logger.info('+' * 100)

            # controlliamo se dobbiamo skippare della lingue e modalità
            if info:
                # controlliamo se siamo dentro alla lingua corretta
                if language['name'] != info['language']:
                    continue
                # controlliamo se siamo dentro alla modalità corretta
                if modality != info['modality']:
                    continue
            
            # aggiungiamo system prompt al secondo modello
            if modality == 'edge_why':
                client_reply_msg.add_system_prompt(REPLY_NOT_GOOD_SYSTEM_PROMPT)
            else:
                client_reply_msg.add_system_prompt(REPLY_GOOD_SYSTEM_PROMPT)
            
            start_time_modality = time.time()
            i = 0
            with tqdm(total=row_2_create) as pbar:
                # se dobbiamo raggiungere una riga saltiamo a quella
                if info:
                    i = info['current_idx']
                    pbar.update(i)
                    selected_rows = info['selected_rows']
                """ existing_rows = len(data.loc[(data['LANGUAGE'] == language['name']) & (data['TYPE_EVAL'] == modality)])
                if existing_rows > 0:
                    i = existing_rows
                    pbar.update(existing_rows) """

                while i < row_2_create:
                    # cerchiamo un indice da utilizzare
                    while(1):
                        random_index = labels.sample(n=1).index[0]
                        if random_index  not in selected_rows:
                            break
                    row = labels.iloc[random_index].copy()
                    # rimuoviamo dalla ricerca la riga
                    selected_rows.append(int(random_index))

                    # creiamo la maschera a seconda dei campi disponibili
                    # cerchiamo per ogni campo e salviamo solo le ore non nulle
                    if modality == 'normal':
                        masks_weights = {'ORE_ORDINARIE': 10,'ORE_STRAORDINARIE':1, 'ORE_VIAGGIO':1, 'NOTE':10, 'INEFFICIENCY':10}
                        mods = {'show': 5}
                    elif modality == 'normal_w_correction':
                        masks_weights = {'ORE_ORDINARIE': 10,'ORE_STRAORDINARIE':1, 'ORE_VIAGGIO':1, 'NOTE':10, 'INEFFICIENCY':10}
                        mods = {'mod': 3,'show': 5}
                    else:
                        masks_weights = {'ORE_ORDINARIE': 5,'ORE_STRAORDINARIE':5, 'ORE_VIAGGIO':5, 'NOTE':1, 'INEFFICIENCY':5}
                        mods = {'mod': 10,'show': 1}

                    if row['ORE_ORDINARIE'] == 0:
                        masks_weights.pop('ORE_ORDINARIE')
                    if row['ORE_STRAORDINARIE'] == 0:
                        masks_weights.pop('ORE_STRAORDINARIE')
                    if row['ORE_VIAGGIO'] == 0:
                        masks_weights.pop('ORE_VIAGGIO')
                    if row['INEFFICIENCY'] == 0 or pd.isna(row['INEFFICIENCY']):
                        masks_weights.pop('INEFFICIENCY')
                    if pd.isna(row['NOTE']):
                        masks_weights.pop('NOTE')

                    # verifichiamo se possiamo lavorare con questa riga: se no proviamo con un altra riga
                    if len(masks_weights) < 2:
                        # non incrementiamo per mantenere la coerenza
                        continue
                    
                    

                    
                    # creiamo l'object
                    row_to_add = generate_msg(row, client_first_msg, client_reply_msg, client_summary, masks_weights, modality=modality, mods=mods, language=language['name'], frasi=language['frasi'])
                    
                    # calcoliamo l'indice 
                    #current_i = len(data) + len(tmp_rows)
                    # inseriamo anche l'indice 
                    #row_to_add['INDICE'] = current_i
                    
                    # inseriamo l'indice 
                    row_to_add['id'] = f"multi-turn-conversation-{language['name']}-{modality}-{i}"
                    # inseriamo la reference sul dataset originale 
                    row_to_add['reference_id'] = int(random_index)

                    # inseriamo i tools
                    row_to_add['agent_tools'] = [{"name": "extractor_expert", "description": "Extract and structure work-related information from user input", "parameters": {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string", "description": "A brief summary of the work activity mentioned by the user"}}}}, {"name": "push_data", "description": "Push the data to a db", "parameters": {"type": "object", "required": ["push"], "properties": {"push": {"type": "boolean", "description": "A boolean that confirm the data will need to push to the db"}}}}]
                    row_to_add['extractor_tool'] = [{"name": "extractor_api", "description": "File per il modello Pydantic da utilizzare per avere uno structured output per l'estrattore delle informazioni", "properties": {"ore_ordinarie": {"description": "ore di lavoro ordinarie effettuate", "title": "Ore Ordinarie", "type": "number"}, "ore_straordinarie": {"description": "ore di lavoro straordinarie effettuate", "title": "Ore Straordinarie", "type": "number"}, "ore_viaggio": {"description": "ore di viaggio effettuate", "title": "Ore Viaggio", "type": "number"}, "durata_inefficienza": {"description": "inefficienze riscontrata durante il lavoro, se si trova settare le ore altrimenti deve essere 0", "title": "Durata Inefficienza", "type": "number"}, "note_inefficienza": {"description": "le note usate per descrivere il tipo di inefficienza riscontrato, se non si trova settare a null", "title": "Note Inefficienza", "type": "string"}, "note": {"description": "usalo per inserire le cause per cui non ci sono state ore di lavoro oppure altre informazioni sulla giornata, deve essere breve e conciso senza l'utilizzo di tempi verbali", "title": "Note", "type": "string"}, "commessa": {"description": "numero della commessa scritto nel messaggio, se non compare sarà fissato a null", "title": "Commessa", "type": "string"}}}]
                    # incrementiamo la progress bar
                    i += 1
                    pbar.update(1)

                    logger.debug(f"aggiungiamo la riga: {row_to_add}")
                    # appendiamo nel file
                    with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
                        writer.write(row_to_add)

                    if i % run_schema['break_step'] == 0:
                        logger.info(f'facciamo riposare per 30s')
                        # nel mentre salviamo anche i checkpoint del run attuale
                        save_current_run_info(i,language['name'], modality, selected_rows)

                        
                        for _ in tqdm(range(30)):
                            time.sleep(1)
                        logger.info("\nriprendiamo la crazione\n")

            # nel mentre salviamo anche i checkpoint del run attuale
            save_current_run_info(i,language['name'], modality, selected_rows)

            diff = time.time() - start_time_modality
            logger.info("abbiamo completato il processing della modalità: {} per la linuga: {} in {:.2f} minuti\n".format(modality,language['name'],diff/60))
            # usciamo per cambiare config
            return 
            # inizializziamo il file di checkpoint ma non la lista da indicizzare
            info = None
        
        # visto che abbiamo finito la lingua, ri inizializziamo la lista di indici da utlizzare  
        selected_rows = []
    
    
    diff = time.time() - start_time
    logger.info("abbiamo completato il processing del dataset in {:.2f} minuti\n".format(diff/60))

    

if __name__ == "__main__":
    Preprocessing()