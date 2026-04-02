from models.OllamaHttpClient import OllamaClient
from models.ModelProviderClient import clientRouter
from utils.logger import Logger
from omegaconf import OmegaConf
import pandas as pd
from schema.db import DB_Vector, DB_Messaggi
from schema.postgres_models import Etichetta
from tqdm import tqdm
from utils.util import remove_single_quotes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from time import sleep
import pyodbc

# get config file
conf = OmegaConf.load("config.yaml")


logger = Logger(save=False, consoleLevel="INFO").getLogger()
def testEmbed_w_ollama(model, db:DB_Vector, test_set:pd.DataFrame=None):
    """
    Test embedding with ollama
    """
    correct = 0
    if  not test_set.empty:
        logger.info(f"testiamo {len(test_set)} queries")
        for idx in tqdm(range(len(test_set))):
            text = test_set.iloc[idx]['HOURS_BY_CLIENT']
            # clean del testo
            text = remove_single_quotes(text)

            embedding_response = model.embed(prompt=text, dim=conf.database.vector_dim)
            embed_query = embedding_response['embeddings'][0]['embedding']
            # facciamo il search per similarità 
            search_result = db.find_similar(table_name=conf.database.postgres_vc_table, query_emb=embed_query, method='l2')
            # confrontiamo solo la prima label
            if search_result.iloc[0]['etichetta'] == test_set.iloc[idx]['INEFFICIENCY_TYPE']:
                correct += 1
        
        logger.info("l'accuracy finale del test set è {:.2f} ".format(correct/len(test_set)))
    else:
        # search a similar query
        query = "corso di sicurezza fino le 10.30, ricerca materiale piazzati lontani dal posto di montaggio, identificazione materiale messi al aperto e senza identificazione pero motivi di pioggia"
        # embed
        embedding_response = model.embed(prompt=query, dim=conf.database.vector_dim)
        embed_query = embedding_response['embeddings'][0]

        """ search_result = clientQdrant.query_points(
            collection_name=collection_name,
            query=embed_query[0],
            with_payload=True,
            limit=3
        ).points """
        # facciamo il search per similarità 
        search_result = db.find_similar(table_name=conf.database.postgres_vc_table, query_emb=embed_query, method='l2')
        logger.info('risulati trovati per la query:')
        logger.info(search_result)

    #search_result.to_csv('prova.csv')



def createVecDB(model, dataset_path:str, test:bool=False):
    """Creazione della tabella eticette"""
    etichette_db = DB_Vector(conf)

    #etichette_db.add_table(table_name=conf.database.postgres_vc_table)

    #etichette_db.drop_table(table=conf.database.postgres_vc_table)

    #etichette_db.add_table(table_name=conf.database.postgres_vc_table)

    # aggiungiamo le righe nel db
    dataset = pd.read_csv(dataset_path)

    if test:
        # splittiamo tra train e test 
        train_set = dataset.sample(frac=0.8, random_state=42)
        test_set = dataset.drop(train_set.index)

        for idx in tqdm(range(len(train_set))):
            text = train_set.iloc[idx]['HOURS_BY_CLIENT']
            # clean del testo
            text = remove_single_quotes(text)

            embedding_response = model.embed(prompt=text, dim=conf.database.vector_dim)
            embedding = embedding_response['embeddings'][0]['embedding']

            # creiamo prima l'etichetta     
            obj = Etichetta(etichetta=train_set.iloc[idx]['INEFFICIENCY_TYPE'], testo=text, embedding=embedding)
            etichette_db.insert(table_name=conf.database.postgres_vc_table,row=obj)
    else:
        # aggiungiamo al vc db
        for idx in tqdm(range(len(dataset))):
            text = dataset.iloc[idx]['HOURS_BY_CLIENT']
            # clean del testo
            text = remove_single_quotes(text)

            embedding_response = model.embed(prompt=text, dim=conf.database.vector_dim)
            embedding = embedding_response['embeddings'][0]['embedding']

            # creiamo prima l'etichetta     
            obj = Etichetta(etichetta=dataset.iloc[idx]['INEFFICIENCY_TYPE'], testo=text, embedding=embedding)
            etichette_db.insert(table_name=conf.database.postgres_vc_table,row=obj)
        
    return etichette_db,test_set


def remove_similar(df:pd.DataFrame, soglia=0.90):
    """ Funzione per rimuovere elementi simili tramite il testo """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(list(df['HOURS_BY_CLIENT']))

    tenere = []
    #df_filter = pd.DataFrame(columns=df.columns)
    used = set()

    for i in range(len(df)):
        if i in used:
            continue
        tenere.append(df.iloc[i])
        #df_filter = pd.concat([df_filter,df.iloc[i]], ignore_index=True)

        # find similars
        similarity = cosine_similarity(vectors[i:i+1], vectors).flatten()
        simili = [j for j,sim in enumerate(similarity) if sim > soglia and j != i]
        #logger.debug(simili)
        if simili:
            for sim in simili:
                #logger.debug(f"testo simile almeno al {soglia}% tra:\n {df.loc[i,"HOURS_BY_CLIENT"]}\n {df.loc[sim,"HOURS_BY_CLIENT"]}")
                #sleep(10)
                used.add(sim)
    return pd.DataFrame(tenere)

def cluster_selection(data,model,target):
    """Funzione per filtrare i dati in un sottoinsieme simile a quello originale con Kmeans"""
    if len(data) > target:
        embeddings_dict = model.embed(prompt=list(data['HOURS_BY_CLIENT']), dim=conf.database.vector_dim)
        embeddings = []
        for emb in embeddings_dict['embeddings']:
            embeddings.append(emb['embedding'])
        kmeans = KMeans(n_clusters=target, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # seleziona rappresentati 
        tenere = []
        for i in range(target):
            idx_cluster = [j for j,l in enumerate(labels) if l == i]
            # trovato il centroide del cluster
            centroide = kmeans.cluster_centers_[i]
            # selezioniamo solo il candidato più vicino al centroide
            selected_idx = [embeddings[j] for j in idx_cluster]
            dis = cdist([centroide], selected_idx)[0]
            idx_choose = idx_cluster[dis.argmin()]
            tenere.append(data.iloc[idx_choose])
    return pd.DataFrame(tenere)

def check_coverage(df_originale, df_filtrato, model):
    """Funzione per controllare la diversità """
    emb_orgin_dict =  model.embed(prompt=list(df_originale['HOURS_BY_CLIENT']), dim=conf.database.vector_dim)
    emb_filter_dict =  model.embed(prompt=list(df_filtrato['HOURS_BY_CLIENT']), dim=conf.database.vector_dim)
    emb_org = []
    emb_filter = []
    for emb in emb_orgin_dict['embeddings']:
        emb_org.append(emb['embedding'])
    for emb in emb_filter_dict['embeddings']:
        emb_filter.append(emb['embedding'])
    # matrice di similarità
    sim_metrix = cosine_similarity(emb_org, emb_filter)
    cover = (sim_metrix.max(axis=1) > 0.8).mean()
    logger.info(f"Coverage: {cover:.2%}")
    
def clean_dataset(file_path, model, target):
    """Funzione per pulire i datset prima di creare il vector db"""
    data_original = pd.read_csv(file_path)
    # selezioniamo solo la specifica catergory
    data = data_original.loc[data_original['INEFFICIENCY_TYPE'] == "CUSTOMER ISSUE"].copy()
    logger.info(f"dati originali {len(data)}")
    # elimina duplicati
    data.drop_duplicates(subset=['HOURS_BY_CLIENT'], inplace=True)
    logger.info(f"dati senza duplicati {len(data)}")
    # elimina i dati simili a livello di testo
    soglia=0.90
    data = remove_similar(data, soglia=soglia)
    logger.info(f"dati senza elementi simili al {soglia}% {len(data)}")
    # usiamo il cluster con rappresentanti per elimare
    data = cluster_selection(data,model,target)
    logger.info(f"dopo il cluster: {len(data)}")
    # facciamo il check della coverage
    check_coverage(data_original.loc[data_original['INEFFICIENCY_TYPE'] == "CUSTOMER ISSUE"], data, model)
    # aggiungiamo solo queste al df finale
    # droppiamo tutti prima
    idx_delete = data_original.loc[data_original['INEFFICIENCY_TYPE'] == "CUSTOMER ISSUE"].index
    data_original.drop(idx_delete, inplace=True)
    # concateniamo 
    data_original = pd.concat([data_original, data], ignore_index=True)
    # salviamo 
    data_original.to_csv("./data/second_test/dati_ineff_clean_customer_issue.csv", index_label=False)

def test_SQL_ServerDB():
    """Funzione per testare la connessione è la procedura su sql server"""
    conn = pyodbc.connect(conf.database.sqlserver_conn, autocommit=True)
    
    cursor = conn.cursor()
    query = "EXEC SP_UPD0001_Aggiornamento_righe_intervento  200, 'INT25_10000', '1767', '2025-09-21T00:00:00+02:00', 8.00, 0.00, 2.00, 0, None, None"
    cursor.execute(query)
    # una PRINT in una stored procedure mostra i risulati in un campo messages:
    if cursor.messages:
        ris = cursor.messages[0][1] # [0] il messaggio della exec, [1] testo
        logger.info(f"Il risultato dal SQLServer è: {ris}")


if __name__ == "__main__":
    """
    Test vector DB in postgres and embedding in Ollama 
    """
    
    #emedding_model = clientRouter(conf, name="embedding_model")
    #clean_dataset("./data/second_test/dati_ineff_clean.csv", emedding_model, target=100)
    #db,test_set = createVecDB(emedding_model, "./data/second_test/dati_ineff_clean_customer_issue.csv")

    #test_SQL_ServerDB()

