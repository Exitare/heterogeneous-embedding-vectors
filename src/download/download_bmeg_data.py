import gripql
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

save_folder = Path("data", "bmeg")

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Download bmeg data')
    parser.add_argument("--creds", "-c", type=Path)
    parser.add_argument("--cancer", type=str, choices=["BRCA", "BLCA", "THCA", "STAD", "LAML"], required=True)

    args = parser.parse_args()

    credentials = args.creds
    cancer = args.cancer

    save_folder = Path(save_folder, cancer)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    conn = gripql.Connection("https://bmeg.io/api/", credential_file=credentials)
    G = conn.graph("rc5")

    c = G.query().V(f"Project:TCGA-{cancer}").out("cases").out("samples").as_("sample")
    c = c.out("aliquots").out("gene_expressions").as_("exp")
    c = c.render(
        ["$sample._data.gdc_attributes.submitter_id", "$exp._data.values"])

    print("Downloading data...")
    data = {}
    for row in c.execute(stream=True):
        data[row[0]] = row[1]

    samples = pd.DataFrame(data).T.fillna(0.0)
    print(samples.head())
    print(samples.shape)
    print(f"Saving data to folder {save_folder}...")
    # remove "Unnamed: 0" column
    samples = samples.reset_index().drop(columns="index")
    samples.to_csv(Path(save_folder, "data.csv"), index=True)
