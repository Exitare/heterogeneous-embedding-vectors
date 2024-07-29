import json
import requests
import argparse
import logging
import time
from tqdm import tqdm
from pathlib import Path

# added filter to param to subset by project - code via:
# https://github.com/bmeg/bmeg-etl/blob/develop/transform/gdc/gdc-scan.py

URL_BASE = "https://api.gdc.cancer.gov/"
TOKEN = None
client = requests


# study = "TCGA-ACC"


def query_gdc(endpoint, params):
    """
    query_gdc makes a query to the GDC API while handling common issues
    like pagination, retries, etc.

    The return value is an iterator.
    """
    # Copy input params to avoid modification.
    params = dict(params)
    page_size = 100
    params['size'] = page_size

    # With a GET request, the filters parameter needs to be converted
    # from a dictionary to JSON-formatted string
    if 'filters' in params:
        params['filters'] = json.dumps(params['filters'])

    headers = None
    if TOKEN is not None:
        headers = {
            "X-Auth-Token": TOKEN
        }
    failCount = 0
    # Iterate through all the pages.
    with tqdm(total=page_size) as pbar:
        while True:
            try:
                req = client.get(URL_BASE + endpoint, params=params, headers=headers)
                data = req.json()

                if 'data' not in data:
                    print("Bad return %s" % (data))
                    failCount += 1
                    if failCount >= 10:
                        raise Exception("Too many failures")
                    time.sleep(10)
                else:
                    failCount = 0
                    data = data['data']
                    hits = data.get("hits", [])
                    if len(hits) == 0:
                        return
                    for hit in hits:
                        yield hit
                    pbar.total = data['pagination']['total']
                    pbar.update(data['pagination']['count'])
                    # Get the next page.
                    params['from'] = data['pagination']['from'] + page_size
            except Exception as e:
                if failCount >= 10:
                    logging.warning(str(e))
                    logging.warning(json.dumps(params))
                    raise
                failCount += 1
                print("Connection Issue %s" % (e))
                time.sleep(10)


def scrapeFiles(outfile):
    parameters = {'expand': ",".join(
        ["cases", "cases.aliquot_ids", "cases.project", "cases.samples.portions.analytes.aliquots", "index_files",
         "analysis.metadata.read_groups"]),
        'filters': {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [
                            "TCGA-BRCA",
                            "TCGA-BLCA",
                            "TCGA-LAML",
                            "TCGA-STAD",
                            "TCGA-THCA",
                            "TCGA-COAD"
                        ]
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "files.data_format",
                        "value": "PDF"
                    }
                }
            ]
        }}

    filesOut = open(outfile, "w")
    filesOut.write("[\n")  # Start the JSON array

    first = True
    for row in query_gdc("files", parameters):
        if not first:
            filesOut.write(",\n")
        filesOut.write(json.dumps(row))
        first = False

    filesOut.write("\n]")  # End the JSON array
    filesOut.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--endpoint", default=URL_BASE)
    parser.add_argument("-t", "--token", default=None)
    parser.add_argument("--destination", "-d", required=False, help="Destination file to write",
                        default=Path("data", "annotations"))

    args = parser.parse_args()

    destination = Path(args.destination)
    if not destination.exists():
        destination.mkdir(parents=True)

    destination = Path(destination, "manifest.json")

    URL_BASE = args.endpoint
    if args.token is not None:
        with open(args.token, "rt") as handle:
            TOKEN = handle.read().strip()

    scrapeFiles(destination)
