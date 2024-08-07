import gripql
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

save_folder = Path("data")

if __name__ == '__main__':
    parser = ArgumentParser(description='Download bmeg data')
    parser.add_argument("--creds", "-c", type=Path)
    parser.add_argument("--cancer", type=str, choices=["BRCA", "BLCA", "THCA", "STAD", "LAML", "COAD"], required=True)

    args = parser.parse_args()

    credentials = args.creds
    cancer = args.cancer
    version = args.version

    save_folder = Path(save_folder, "rna")

    save_folder = Path(save_folder, cancer)
    print(f"Using save folder: {save_folder}")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    conn = gripql.Connection("https://bmeg.io/api", credential_file=credentials)
    G = conn.graph("rc5")

    c = G.query().V(f"Project:TCGA-{cancer}").out("cases").out("samples").as_("sample")
    c = c.out("aliquots").out("gene_expressions").as_("exp")
    c = c.render(
        ["$sample._data.gdc_attributes.submitter_id", "$exp._data.values"])

    print("Downloading data...")
    data = {}
    for row in c.execute(stream=True):
        submitter_id = row[0]
        values = row[1]
        if version == "1":
            data[submitter_id] = values
        else:
            if submitter_id not in data:
                data[submitter_id] = {}
            for gene, value in values.items():
                data[submitter_id][gene] = value

    samples = pd.DataFrame.from_dict(data, orient='index').fillna(0.0)
    samples.reset_index(drop=False, inplace=True)
    # rename index column to patient
    samples.rename(columns={"index": "Patient"}, inplace=True)

    print(samples.head())
    print(samples.shape)
    print(f"Saving data to folder {save_folder}...")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    samples.to_csv(Path(save_folder, "data.csv"), index=True)
    print("Done.")
