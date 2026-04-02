import re
def clean_response(text:str):
    """Pulisce la response del modello per ritornare unicamente un json"""
    # pulisci se è presente il tag di thinking ( con il flag re.DOTALL attivo si permette di avere un match anche con una new line \n)
    clean_output = re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL)
    # rimuovi i markdown code blocks
    clean_output = re.sub(r"```json\s*","", clean_output)
    clean_output = re.sub(r"```\s*","",clean_output)
    # rimuovi eventuali whitespace all'inizio e fine
    return clean_output.strip()


def remove_single_quotes(text:str):
    """
    Funzione che pulisce tutti i tipi di single quote da un testo.
    In particolare si rimuove:
    Apostofo normale '(unicode 39)
    Apostofo sinistro '(unicode 8216)
    Apostofo destro '(unicode 8217)
    Apostofo modifier ʼ(unicode 700)
    """
    apostrofi = [chr(8217), chr(39), chr(8216), chr(700)]
    for apostrofo in apostrofi:
        text = text.replace(apostrofo, "")
    
    return text


def process_tel_number(tel:str)->tuple:
    """Funzione per processare il numero di telefono e trovando in che lingua parlare"""
    # se il numero non esiste 
    if tel == None:
        return 'errore', 'numero non presente'
    # rimuoviamo eventuali caratteri non numerici
    clean_number = ""
    language = ""
    for c in tel:
        if c.isdigit():
            clean_number += c
    
    if len(clean_number) == 0:
        return 'errore','numero non presente'
    
    # controlliamo a che stato appartiene. Prima prendiamo solo i primi 3 caratteri
    match clean_number[:2]:
        case '39': # numero italiano
            language = "it"
        case '20': # numero egiziano
            language = "en"
        case '34': 
            # bisogna controllare se si tratta del numero dalla spagna oppure di un numero italiano 340,ecc...
            if len(clean_number) == 10:
                # italiano. dobbiamo aggiungere il prefisso
                clean_number = '39' + clean_number
                language = "it"
            else:
                language = "es"
        case '91': # numero indiano
            language = 'en'
        case '52': # numero messicano
            language = 'es'
        case '60': # numero malesiano
            language = 'en'
        case '35': # forse numero portoghese
            if clean_number[2] == '1':
                language = 'es'
        case '21': # forse numero algerino
            if clean_number[2] == '3':
                language = 'en'
        case '86': # numero cinese 
            language = 'en'
        case '90': # numero turchia
            language = 'en'
        case '97': # numero emirati arabi
            language = 'it'
        case _:
            # supponiamo che sia numero italiano in cui è necessario aggiungere il prefisso
            clean_number = '39' + clean_number
            language = 'it'
    
    return language, clean_number

    

