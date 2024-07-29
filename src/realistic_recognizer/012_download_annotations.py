import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

URL = 'https://api.gdc.cancer.gov/data/'
MAX_RETRIES = 3


def download_file(case_id: str, submitter_id: str):
    file_name = f"{case_id}_{submitter_id}"
    file_path = Path("data", "annotations", f"{file_name}.pdf")
    if file_path.exists():
        print(f"{file_name}.pdf already exists. Skipping download.")
        return

    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            request = requests.get(URL + case_id)
            if request.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(request.content)
                # print(f"Successfully downloaded {case_id}")
                return
            else:
                print(f"Failed to download {case_id} (status code: {request.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading {case_id}: {e}")

        attempts += 1
        if attempts < MAX_RETRIES:
            print(f"Retrying download for {case_id} (attempt {attempts + 1} of {MAX_RETRIES})")
            time.sleep(2)  # Optional: add delay between retries

    print(f"Failed to download {case_id} after {MAX_RETRIES} attempts.")


def main():
    annotations = pd.read_json(Path("data", "annotations", "manifest.json"))
    case_ids = annotations["id"]
    # get all submitter ids
    submitter_ids = [case[0]["submitter_id"] for case in annotations["cases"]]
    combined_ids = list(zip(case_ids, submitter_ids))

    # assert that combined ids are the same as case ids and submitter ids
    assert combined_ids[0][0] == case_ids[0] and combined_ids[0][1] == submitter_ids[
        0], "Combined ids should be the same as case ids and submitter ids"
    # assert that first case id containts first submitter id

    assert combined_ids[0][1] == annotations["cases"][0][0][
        "submitter_id"], "First case id should contain first submitter id"

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_file, case_id, submitter_id): (case_id, submitter_id) for
                   case_id, submitter_id in combined_ids}

        for future in tqdm(as_completed(futures), total=len(case_ids)):
            future.result()  # Will raise an exception if the download failed


if __name__ == "__main__":
    main()
