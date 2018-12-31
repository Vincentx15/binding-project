# 3D shape embedder and benchmark on the DUDE

## Embedding
The embedding interface is implemented in the moment computer class and called in the csv builder
USRCAT and USRPCAT implement the two techniques. The fine tuning of the parameters are propagated and can be called from the main, explained in the corresponding file

## Enrichment Factor
The benchmarking is then conducted in the EF_computer file. It will produce one csv for each experiment that will get saved in the data/raw_output_csv. These results are then aggregated in the post_processor file