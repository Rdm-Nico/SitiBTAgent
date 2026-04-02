import os
import pandas as pd
from omegaconf import OmegaConf
from utils.logger import Logger
from models.OllamaHttpClient import OllamaClient
from utils.Tool import ExtractorTool, PushTool
from tqdm import tqdm
from schema.structured_ouput_model import Tecnico
from utils.util import clean_response
from time import sleep
import argparse
import json
CSV_LABLES="./data/second_test/labels/labels_edge_flow.csv"
MODELS={
        'qwen3': ('chat-qwen3:4b-instruct', 'extractor-qwen3:4b-instruct'),
        'granite4': ('chat-granite4:latest', 'extractor-granite4:latest')
        }
OUTPUT_FILE="./data/second_test/results/results_edge_flow_2.csv"
COLUMNS = [
            'indice','model','language','messaggi', 'ore_ordinarie_true', 'ore_straordinarie_true', 'ore_viaggio_true', 'note_true', 'inefficiency_true','inefficiency_note_true',
           'chat_context','context_window', 'function_calls','push_data_response','ore_ordinarie_acc', 'ore_straordinarie_acc', 'ore_viaggio_acc',
            'inefficiency_acc','tot_acc', 'total_duration', 'load_duration_chat', 'prompt_eval_duration_chat', 'eval_duration_chat', 'eval_count_chat',
            'load_duration_extractor', 'prompt_eval_duration_extractor', 'eval_duration_extractor', 'eval_count_extractor',
            'ore_ordinarie_predict', 'ore_straordinarie_predict', 'ore_viaggio_predict', 'note_predict', 'inefficiency_predict','inefficiency_note_predict',
            'REF_INDICE','L_MESSAGGI','TYPE_EVAL','W_MESSAGGI'
        ] 

SAVE_CHECKPOINTS = 50 
RESUME_CHECKPOINTS = -1

logger = Logger(save=False, consoleLevel="INFO").getLogger()
conf = OmegaConf.load("config.yaml")

def load_data():
    # open labels folder and iterate on every file inside
    datasets = pd.read_csv(CSV_LABLES) 
    return datasets



def Test(runner_args):
    """
    Test con il dataset da 767 righe  
    """
    logger.info(f"Runner Settings args:\n{runner_args}")
    # prepariamo il file di output 
    output_path = f"./data/second_test/results/{runner_args.output_filename}.csv"
    # load test sets
    datasets = load_data()
    logger.info(f'dataset_path {CSV_LABLES}\t data returned:{datasets.shape}\t Columns: {datasets.columns}')
    logger.info(f"dateset  label's distribution:\n {datasets.groupby(['TYPE_EVAL'])['L_MESSAGGI'].count()}")

    # inserisci 0 nella colonna delle inefficienze se non è presente un valore
    datasets.fillna(value={"INEFFICIENCY": 0}, inplace=True)


    logger.info(f"fatto il load del dataset: {len(datasets)} righe da valutare")
    # load ollama
    chat_client = OllamaClient(conf)
    extractor_client = OllamaClient(conf)
    chat_client.add_tools(tools=[ExtractorTool(), PushTool()])

    if runner_args.ollama_host:
        # cambio del base url di Ollama
        chat_client.add_hostname(runner_args.ollama_host)
        extractor_client.add_hostname(runner_args.ollama_host)


    # init the dataframe or open the created one for save the scores
    results = pd.DataFrame()
    if os.path.exists(output_path):
        results = pd.read_csv(output_path)
        logger.info(f"riprendiamo dal file {output_path}")
    else:
        results = pd.DataFrame(columns=COLUMNS)

    # for each model
    for idx_model,model_name in enumerate(MODELS.keys()):
        chat_model, extractor_model = MODELS[model_name]
        logger.info('-' * 100)
        logger.info(f'modello da testare {model_name}\n')
        logger.info('-' * 100)

        chat_client.add_model(chat_model)
        extractor_client.add_model(extractor_model)

        length = len(datasets)
        i = 0
        with tqdm(total=len(datasets)) as pbar:
            if len(results) > 0:
                # controlliamo a che parte siamo 
                i = len(results) - idx_model*len(datasets)
                pbar.update(i)
            
            # lista temporanea per le nuove righe
            tmp_rows = []
            while i < length:

                result = {}
                result['model'] = model_name
                row = datasets.iloc[i]
                # indice di riferimento dai dati: ./data/second_test/raw_data_combinated_with_less_inf.csv
                result['REF_INDICE'] = row['REF_INDICE']
                messaggi_str = row['MESSAGGI']
                # input da tenere
                result['language'] = row['LANGUAGE']
                result['ore_ordinarie_true'] = row['ORE_ORDINARIE']
                result['ore_straordinarie_true'] = row['ORE_STRAORDINARIE']
                result['ore_viaggio_true'] = row['ORE_VIAGGIO']
                result['inefficiency_true'] = row['INEFFICIENCY']
                result['inefficiency_note_true'] = row['HOURS_BY_CLIENT']
                result['note_true'] = row['NOTE']
                result['messaggi'] = messaggi_str
                result['L_MESSAGGI'] = row['L_MESSAGGI']
                result['TYPE_EVAL'] = row['TYPE_EVAL']
                result['W_MESSAGGI'] = row['W_MESSAGGI']


                result['function_calls'] = []
                result['total_duration'] = 0
                result['prompt_eval_duration_chat'] = 0
                result['load_duration_chat'] = 0
                result['eval_duration_chat'] = 0
                result['eval_count_chat'] = 0
                result['prompt_eval_duration_extractor'] = 0
                result['load_duration_extractor'] = 0
                result['eval_duration_extractor'] = 0
                result['eval_count_extractor'] = 0


                # convertiamo la stringhe in messaggi
                messaggi = eval(messaggi_str)
                # la risposta del modello
                chat_response = None
                # il nostro flusso di messaggi sarà: msg_init + N(msg_inter) + msg_finale
                for num,msg in enumerate(messaggi):

                    chat_response = chat_client.chat(message=msg, save_to_history=True)
                    logger.debug(f"messaggio inviato({num}/{len(messaggi)}): {msg}")

                    # controlliamo se abbiamo ricevuto una risposta valida
                    if 'eval_duration' in chat_response:
                        # save times
                        try:
                            result['total_duration'] += chat_response['total_duration'] / 10**9
                            result['prompt_eval_duration_chat'] += chat_response['prompt_eval_duration'] / 10**9
                            result['load_duration_chat'] += chat_response['load_duration'] / 10**9
                            result['eval_duration_chat'] += chat_response['eval_duration'] / 10**9
                            result['eval_count_chat'] += chat_response['eval_count']
                        except KeyError as e:
                            raise Exception(f"errore per la riga {i} e il messaggio '{msg}' la response è : {chat_response}.\n errore: {str(e)}")

                    # guardiamo se ha chiamato la funzione:
                    if 'message' in chat_response and 'tool_calls' in chat_response['message']:
                        funciton_name = chat_response["message"]['tool_calls'][0]['function']['name']
                        result['function_calls'].append(funciton_name)

                        if funciton_name == 'extractor_expert':
                            summary = chat_response["message"]['tool_calls'][0]['function']['arguments']['summary']

                            extractor_response = extractor_client.generate(prompt=summary, schema=Tecnico)
                            # save times
                            result['total_duration'] += extractor_response['total_duration'] / 10**9
                            result['prompt_eval_duration_extractor'] += extractor_response['prompt_eval_duration'] / 10**9
                            result['load_duration_extractor'] += extractor_response['load_duration'] / 10**9
                            result['eval_duration_extractor'] += extractor_response['eval_duration'] / 10**9
                            result['eval_count_extractor'] += extractor_response['eval_count']


                            clean_output = clean_response(extractor_response['response'])
                            try:
                                clean_json = json.loads(clean_output)


                                predict = Tecnico.model_validate_json(json.dumps(clean_json))
                            except Exception as e:
                                # salviamo prima i risultati fino ad ora 
                                if tmp_rows:
                                    new_df = pd.DataFrame(tmp_rows)
                                    results = pd.concat([results, new_df], ignore_index=True)
                                    # svuotiamo la lista
                                    tmp_rows = []
                                # sistemiamo l'indice per sicurezza
                                results['indice'] = results.index
                                results.to_csv(output_path, index=False)
                                raise Exception({"error": f"Errore: {str(e)}\n model_response: {extractor_response['response']},\n clean_output: {clean_output}"})

                            try:
                                # calcoliamo l'accuracy solo nelle labels che sono diverse da zero.
                                denominator = 0
                                accumulator = 0
                                if row['ORE_ORDINARIE'] != 0:
                                    result['ore_ordinarie_acc'] = 1 if predict.ore_ordinarie == row['ORE_ORDINARIE'] else 0
                                    accumulator += 1 if predict.ore_ordinarie == row['ORE_ORDINARIE'] else 0
                                    denominator +=1
                                if row['ORE_STRAORDINARIE'] != 0:
                                    result['ore_straordinarie_acc'] = 1 if predict.ore_straordinarie == row['ORE_STRAORDINARIE'] else 0
                                    accumulator += 1 if predict.ore_straordinarie == row['ORE_STRAORDINARIE'] else 0
                                    denominator +=1
                                if row['ORE_VIAGGIO'] != 0:
                                    result['ore_viaggio_acc'] = 1 if predict.ore_viaggio == row['ORE_VIAGGIO'] else 0
                                    accumulator += 1 if predict.ore_viaggio == row['ORE_VIAGGIO'] else 0
                                    denominator +=1
                                if row['INEFFICIENCY'] != 0:
                                    result['inefficiency_acc'] = 1 if predict.durata_inefficienza == row['INEFFICIENCY'] else 0
                                    accumulator += 1 if predict.durata_inefficienza == row['INEFFICIENCY'] else 0
                                    denominator +=1
                                
                                if denominator == 0:
                                    # significa che dovrebbe essere una giornata di riposo in cui non si indicano le ore. In questo caso controlliamo ogni risultato per non avere dati inventati
                                    result['ore_ordinarie_acc'] = 1 if predict.ore_ordinarie == row['ORE_ORDINARIE'] else 0
                                    result['ore_straordinarie_acc'] = 1 if predict.ore_straordinarie == row['ORE_STRAORDINARIE'] else 0
                                    result['ore_viaggio_acc'] = 1 if predict.ore_viaggio == row['ORE_VIAGGIO'] else 0
                                    result['inefficiency_acc'] = 1 if predict.durata_inefficienza == row['INEFFICIENCY'] else 0
                                    result['tot_acc'] = ( result['ore_viaggio_acc'] + result['ore_straordinarie_acc'] + result['ore_ordinarie_acc'] + result['inefficiency_acc'] ) / 4
                                else:
                                    result['tot_acc'] = accumulator / denominator
                                

                                result['note_predict'] = predict.note
                                result['ore_ordinarie_predict'] = predict.ore_ordinarie
                                result['ore_straordinarie_predict'] = predict.ore_straordinarie
                                result['ore_viaggio_predict'] = predict.ore_viaggio
                                result['inefficiency_predict'] = predict.durata_inefficienza
                                result['inefficiency_note_predict'] = predict.note_inefficienza

                            except KeyError as e:
                                raise KeyError(f" Errore: {str(e)} \n json: {predict}\n model_response: {extractor_response['response']},\n clean_output: {clean_output}")

                            # save in the context the response of the chat model
                            if predict.durata_inefficienza != 0:
                                find_inefficienza = True
                                output = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{find_inefficienza}*\ndurata inefficienza: *{predict.durata_inefficienza}*\netichetta inefficienza: *ETICHETTA*\ndescrizione inefficienza: *{predict.note_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{predict.commessa}*\n\nsono corretti?"
                            else:
                                output = f"\nHo riconosciuto i seguenti campi:\nore effetuate: {predict.ore_ordinarie}\nore straordinarie: {predict.ore_straordinarie}\nore di viaggio: {predict.ore_viaggio}\ncommessa: {predict.commessa}\nnote: {predict.note}\n\nsono corretti?"
                            chat_client.save_2_conversation(output)

                        elif funciton_name == 'push_data':
                            # save chat params of the function 
                            try:
                                ris_push = chat_response["message"]['tool_calls'][0]['function']['arguments']['push']
                                result['push_data_response'] = ris_push
                            except KeyError as e:
                                # è avvenuto un problema in corrispodenza dell'argomento push. Lo settiamo a false
                                result['push_data_response'] = False
                    else:
                        # se non viene chiamato nessun tool dovrebbe essere un errore se non ci troviamo nella parte dei test con frasi random
                        if row['TYPE_EVAL'] != 'edge_random': 
                            if msg == messaggi[0]:
                                # the error accour in the first message -> we put the accuracy to 0
                                result['ore_ordinarie_acc'] = 0
                                result['ore_straordinarie_acc'] =  0
                                result['ore_viaggio_acc'] =  0
                                result['inefficiency_acc'] = 0
                                result['tot_acc'] = 0
                            else: 
                                result['push_data_response'] = False

                logger.debug(f"finito il sample {i}")
                # we save the prompt_eval_count + eval_count here because should rappresent the context window
                if chat_response:
                    result['context_window'] =  chat_response['eval_count'] + chat_response['prompt_eval_count']
                # save all the context 
                result['chat_context'] = chat_client.export_conversation()
                chat_client.clear_history()

                # aggiungiamo l'indice
                current_i = len(results) + len(tmp_rows)

                result['indice'] = current_i
                # appendiamo 
                tmp_rows.append(result)
                # incrementiamo la progress bar
                i += 1
                pbar.update(1)

                if i % SAVE_CHECKPOINTS == 0:
                    logger.info("facciamo riposare 1 minuto la GPU...")
                    # salva il df corrente ad ogni iterazioni
                    if tmp_rows:
                        new_df = pd.DataFrame(tmp_rows)
                        results = pd.concat([results, new_df], ignore_index=True)
                        # svuotiamo la lista
                        tmp_rows = []
                    # sistemiamo l'indice per sicurezza
                    results['indice'] = results.index
                    results.to_csv(output_path, index=False)
                    logger.info(f"risultati salvati al checkpoint {i} in '{output_path}'\n")
                    for _ in tqdm(range(60)):
                        sleep(1)
                    print("\nriprendiamo la crazione\n")
            

        # unload del modello dalla VRAM
        logger.info(chat_client.unload_model()['output'])
        # salva il df corrente ad ogni iterazioni
        if tmp_rows:
            new_df = pd.DataFrame(tmp_rows)
            results = pd.concat([results, new_df], ignore_index=True)
        # sistemiamo l'indice per sicurezza
        results['indice'] = results.index
        results.to_csv(output_path, index=False)
        logger.info(f"risultati salvati in '{output_path}'\n")
        logger.info(f'==================================================\n')



if __name__ == "__main__":
    # add the config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",action="store", dest="ollama_host", help="ollama server address", default=None
    )
    parser.add_argument(
        "--name",action="store", dest="output_filename", help="filename of the output results", required=True
    )
    args = parser.parse_args()
    Test(args)