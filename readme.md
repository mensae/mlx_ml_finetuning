Questa repository è una guida minimale su come effettuare finetuning di un modello LLM locale usando il framework [mlx-lm](https://github.com/ml-explore/mlx-lm).

Questo framework è progettato specificamente per funzionare su MacOS con processori Silicon. 
Se siete interessati a effettuare finetuning su dispositivi Windows o Linux, potete usare la libreria [Unsloth](https://docs.unsloth.ai/get-started/fine-tuning-guide).

# Setup di base

Clonare questa repo:
```
https://github.com/mensae/mlx_ml_finetuning.git
```

Creare un env di Python per il progetto ed entrarci.
```
python -m venv venv
source venv/bin/activate
```

Installare le librerie necessarie dai requirements:
```
pip install -r requirements.txt
```

## Struttura del progetto
```
project_root
│
├── data (contiene il dataset)
│   ├── raw
│   │   ├── messages_v1.json
│   ├── for_train
│	│   ├── v1
│	│	│	├── train.jsonl
│	│   │   ├── valid.jsonl
│
├── models (contiene i diversi modelli)
│   ├── base
│   │   ├── gemma-3-text-27b-it-4bit/
│   ├── adapters
│   │   ├── adapter-v1_gemma-3-text-27b-it-4bit/
│   ├── fused
│   │   ├── fused-v1_gemma-3-text-27b-it-4bit/
│
├── build_train.py
├── fine_tune.sh
├── fuse.sh
├── test_on_validation.py
```
Dato che in questa repository ovviamente non ci sono i modelli né i dati le cartelle data e models sono da creare da parte vostra.

# Fine tuning

## Scaricare modello base

Per scaricare il modello base ricorriamo alla libreria `huggingface-cli`, che permette di scaricare modelli direttamente da [HuggingFace](https://huggingface.co/). Mlx-lm è compatibile con svariati modelli, ma per stare tranquilli sulla compatibilità conviene usare quelli direttamente forniti nella repository di HuggingFace [mlx-community](https://huggingface.co/mlx-community/).

Gli esperimenti di questa repo fanno uso del modello `mlx-community/gemma-3-text-27b-it-4bit` (Gemma 3, 27 miliardi di parametri, instruction tuned, quantizzato a 4bit), ma idealmente si può certamente sostituire con qualunque modello offerto da mlx-community.

Quindi, con questo comando scarichiamo il modello in questione nella directory locale `models/base`:
```
huggingface-cli download mlx-community/gemma-3-text-27b-it-4bit  --local-dir ./models/base/gemma-3-text-27b-it-4bit
```

## Preparare il dataset

Per preparare il dataset per il finetuning dobbiamo generare due set: 
- Training set: necessario per l'allenamento
- Evaluation set: necessario per evitare l'overfitting

A tal fine, sfruttiamo il codice `build_train.py`. Il codice è molto semplice e le operazioni svolte sono:
- Caricare il dataset che si trova in `data/raw`
- Filtrarlo in modo da preservare solo le entries di maggiore qualità.
- Convertirlo nel formato corretto, ossia una lista di entries `{'prompt': '...', 'completion': '...'}`.
- Sfruttare la funzione `train_test_split()` di sklearn per ottenere i file `train.jsonl` e `valid.jsonl`, che saranno piazzati nella cartella `data/for_train`.

## Generare l'adapter
La generazione dell'adapter (ossia il vero e proprio fine tuning) viene richiamata tramite lo script `fine_tune.sh`, che semplicemente richiama `mlx_lm.lora` con i parametri correttamente impostati. Si ponga particolare attenzione al parametro `iters` che permette di determinare quante iterazioni devono essere fatte durante il finetuning e `num-layers` che determina su quanti layer fare il finetuning.

## Fondere i modelli
Per il momento abbiamo generato un adapter, che può essere usato per la generazione ma è scomodo. Si può generare direttamente un modello fuso che incolla l'adapter permanentemente al modello base. Questa operazione è svolta dallo script `fuse.sh`. 

Questa guida è semplificata, ci sono altre impostazioni e utilizzi interessanti che potete leggere [qui](https://github.com/ml-explore/mlx-examples/tree/main/lora). Per esempio è possibile usare `mlx_lm.chat` per dialogare direttamente col vostro modello.

