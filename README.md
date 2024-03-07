# OnChainData

analysis of blockchain data

based on https://github.com/blockchain-etl

## Use Virtual Environment
For first time use. 

```
. script/initialize.sh
```

After first time use
```
. venv/bin/activate
```

## Fetch Data

Fetch Bitcoin
```
python3 -m script.fetch_bitcoin
```

Fetch ETH
```
python3 -m script.fetch_ETH
```

## Analyze Data

Analyze Bitcoin
```
python3 -m src.bitcoin_analysis.main
```

Analyze ETH
```
TODO
```

