from pydantic import BaseModel, Field
class Tecnico(BaseModel):
    """File per il modello Pydantic da utilizzare per avere uno structured output per l'estrattore delle informazioni"""
    ore_ordinarie:float = Field(description="ore di lavoro ordinarie effettuate")
    ore_straordinarie:float = Field(description="ore di lavoro straordinarie effettuate")
    ore_viaggio:float = Field(description="ore di viaggio effettuate")
    durata_inefficienza:float = Field(description="inefficienze riscontrata durante il lavoro, se si trova settare le ore altrimenti deve essere 0")
    note_inefficienza:str = Field(description="le note usate per descrivere il tipo di inefficienza riscontrato, se non si trova settare a null")
    note:str = Field(description="usalo per inserire le cause per cui non ci sono state ore di lavoro oppure altre informazioni sulla giornata, deve essere breve e conciso senza l'utilizzo di tempi verbali")
    commessa:str = Field(description="numero della commessa scritto nel messaggio, se non compare sarà fissato a null")


class FollowUp(BaseModel):
    """File per il modello Pydantic da utilizzare per avere uno structured output per l'estrattore delle informazioni"""
    prob_acquisizione:float = Field(description="Probabilità che il cliente sia pronto a comprare l offerta fatta")
    data_consegna:str = Field(description="data della consegna prevista")
    prezzo_vendita:float = Field(description="prezzo di vendita previsto dell offerta")
    note:str = Field(description="tutte le informazioni riguardanti l'offerta, questo campo deve essere dettagliato e il più simile possibile a quello passato")
    codice_offerta:str = Field(description="codice della offerta , se non compare sarà fissato a null")