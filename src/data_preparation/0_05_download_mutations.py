import pandas as pd
import gripql
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser(description='Download bmeg data')
    parser.add_argument("--creds", "-c", type=Path, help="The path to the bmeg credentials file.", required=True)
    args = parser.parse_args()

    credentials = args.creds

    # get the sheet from https://cancer.sanger.ac.uk/cosmic/census?tier=all
    genesDF = pd.read_csv(Path("data", "mutations", "gene_mutations.tsv"), sep="\t", index_col=0)

    geneList = genesDF[(genesDF["Somatic"] == "yes")].index.to_list()

    conn = gripql.Connection(url='https://bmeg.io/grip', credential_file=credentials)
    G = conn.graph("rc6")
    geneMap = dict((i[0], i[1]) for i in G.query().V().hasLabel("Gene").render(["gene_id", "symbol"]).execute())

    hugoMap = {}
    for k, v in geneMap.items():
        if v in geneList:
            hugoMap[v] = k

    projects = G.query().V('Program:TCGA').out("projects").render("$._gid").execute()

    targetList = list(hugoMap.values())

    data = {}
    for p in tqdm(projects):
        q = G.query().V(p).out("cases").out("samples").as_("sample")
        q = q.has(gripql.eq("$.gdc_attributes.sample_type", "Primary Tumor"))
        q = q.out("aliquots").out("somatic_callsets")
        q = q.outE().has(gripql.within("ensembl_gene", targetList))
        q = q.as_("variant")
        q = q.render(["$sample.submitter_id", "$variant.ensembl_gene"])
        for sample, gene in q:
            if sample not in data:
                data[sample] = {gene: 1}
            else:
                data[sample][gene] = 1

    X = pd.DataFrame(data).rename(index=geneMap).fillna(0).transpose()
    # reset index
    X = X.reset_index()
    # rename the index column
    X = X.rename(columns={"index": "submitter_id"})
    # save the data
    X.to_csv(Path("data", "mutations", "mutations.csv"), index=False)
