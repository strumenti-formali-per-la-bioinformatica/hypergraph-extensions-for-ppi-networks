# Hypergraph Extensions for PPI Networks

![Static Badge](https://img.shields.io/badge/python-3.10.14-green)
![Static Badge](https://img.shields.io/badge/torch-2.0.0%2Bcu121-blue)
![Static Badge](https://img.shields.io/badge/torch_geometric-2.5.0-blue)

Nel corso degli ultimi anni lo sviluppo del *deep learning geometrico* ha portato ad una importante crescita degli studi riguardo le capacità di queste reti neurali nella risoluzione di complessi problemi legati ai grafi e ai sistemi rappresentabili come reti. Tuttavia, nonostante l'elevato potenziale di questi modelli rispetto a quelli tradizionali, la capacità di queste particolari reti neurali, così come per le reti tradizionali, è comunque influenzata dalla qualità e dalla struttura dei dati sui quali lavorano.

In questo lavoro, si andrà a considerare un possibile metodo per arricchire i grafi utilizzando i pattern strutturali nascosti per migliorare le performance di diversi task applicati ad essi; in particolare ci focalizzeremo sulle PPI network, i cui nodi rappresentano delle proteine e gli archi descrivono una interazione o una certa soglia di similarità tra le proteine. Lo scopo è enfatizzare l'importanza dei motif, e più nello specifico di questo lavoro, le clique di diversa taglia, come pattern strutturale per le reti biologiche.

<table border="0">
    <tr>
        <td>
            <img src="assets/multi-layer-network.png" alt="Multi-Layer Network">
        </td>
        <td>
            <img src="assets/hypergraph.png" alt="Multi-Layer Network">
        </td>
    </tr>
</table>

## Installazione

```
python3 -m pip install -r requirements.txts
# Install torch according to your machine specifications
conda install pyg -c pyg
```

## Riproduzione degli esperimenti

Nel caso del task di *function-prediction*, occorre prima di tutto lanciare lo script per scaricare e generare un ipergrafo a partire dal dataset di riferimento.

### Function Prediction

Al fine di trasformare il grafo in un ipergrafo utilizzando la tecnica descritta in questo progetto, eseguire il comando `python3 prepare_dataset.py`, specificando l'indice da utilizzare per la previsione (*jc*, *aa*, *ra*).

Per eseguire gli esperimenti per il task di *function-prediction* utilizzando il modello basato su *Graph Convolutional Network* descritto nella relazione di progetto:

```bash
python3 function_prediction.py --model gcn
python3 function_prediction.py --model hypergcn
```

### Link Prediction

#### Link Prediction PPI24

```bash
python3 link_prediction24.py --model gcn
python3 link_prediction24.py --model hypergcn

# Oppure utilizzando features generate casualmente

python3 link_prediction24.py --random_features --model gcn
python3 link_prediction24.py --random_features --model hypergcn
```

#### Link Prediction PPI147

```bash
python3 link_prediction147.py --model gcn
python3 link_prediction147.py --model hypergcn
```
