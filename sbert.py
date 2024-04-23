from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import argparse

model = SentenceTransformer("all-MiniLM-L6-v2")

save_folder = Path("results", "embeddings")

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    # Our sentences to encode
    sentences = [
        "Breast cancer is currently the most prevalent cancer globally and the leading cause of cancer deaths.",
        "It affects women across both developed and developing nations.",
        "Prevention and early diagnosis are crucial for a favorable outcome.",
        "Cancer cells are characterized by their capability for unlimited division, rendering them immortal.",
        "In normal cells, telomeres shorten with each division, but in cancer cells, telomerase rebuilds them.",
        "Telomerase is expressed in over 85% of all cancers and up to 100% of adenocarcinomas, including breast cancer.",
        "Telomerase may perform various functions, some related to telomeres and others not.",
        "High activity of telomerase in cancer cells correlates with reduced sensitivity to therapies.",
        "Consequently, telomerase has been identified as a potential target for cancer treatments.",
        "The ineffectiveness of current therapies has spurred the search for new, more effective combined treatments.",
        "These include the use of telomerase inhibitors and telomerase-targeted immunotherapy.",
        "Cancer remains one of the most feared diseases of the 20th century, with its prevalence continuing to increase into the 21st century.",
        "Currently, every fourth person has a lifetime risk of developing cancer.",
        "In India alone, more than 1.1 million new cancer cases are reported annually, while globally this number exceeds 14 million.",
        "Cancer can indeed be cured if detected early enough.",
        "Cancer growth can be halted by one of four methods: surgical removal of the tumor, chemotherapy or other cancer-specific medications like hormonal therapy, radiation therapy, or spontaneous regression of cancer cells.",
        "Published data on recent global cancer incidence and mortality trends are limited.",
        "We utilized the International Agency for Research on Cancer's CANCERMondial database to examine age-standardized cancer incidence and death rates from 2003-2007.",
        "We also reviewed trends in incidence up to 2007 and mortality up to 2012 in selected countries from five continents.",
        "High-income countries report the highest incidence of cancers including lung, colorectal, breast, and prostate cancer, although some low- and middle-income countries now also show high rates.",
        "Mortality rates are decreasing in high-income countries due to better risk management, screening, early detection, and improved treatments.",
        "Conversely, in several low- and middle-income countries, mortality rates are rising due to increased smoking, excess body weight, and lack of physical activity.",
        "These countries also face high rates of infection-related cancers.",
        "Effective cancer control measures are urgently needed to manage high rates in high-income countries and curb the increasing burden in low- and middle-income countries.",
        "Breast cancer affects one in seven women worldwide during their lifetime.",
        "Widespread mammographic screening and educational campaigns facilitate early detection, often during the asymptomatic stage.",
        "Current breast cancer management and recurrence monitoring primarily rely on pathological and, increasingly, genomic evaluations of the primary tumor.",
        "Despite extensive research, up to 15% of breast cancer patients experience recurrence within the first 10 years post-surgery.",
        "Local recurrences were initially thought to be caused by tumor cells contaminating histologically normal tissues beyond the surgical margin.",
        "However, technological advances have revealed distinct molecular aberrations in peri-tumoral tissues, supporting the field cancerization theory.",
        "This theory posits that tumors originate from a field of molecularly altered cells that enable malignant evolution, with or without morphological changes.",
        "Our review examines the pathological and molecular features of field cancerization in peri-tumoral tissues, and discusses their implications for research and patient management.",
        "Cancer is a leading cause of death globally, with over 10 million deaths annually.",
        "Current treatments, including surgery, radiation, and chemotherapy, often harm healthy cells and cause significant toxicity.",
        "Research is focused on targeting only cancerous cells due to the intra-tumor heterogeneity that complicates effective treatment.",
        "Advancements in understanding the molecular basis of tumors and new diagnostic technologies are critical for improving cancer treatment.",
        "Studies of epigenetic changes and gene expression in cancer cells are essential, as are techniques to correct or minimize these changes.",
        "Heat Shock Proteins (HSPs) play a crucial role in protein folding and modulate key apoptotic factors.",
        "Their high expression in various cancers, such as breast, prostate, and colorectal cancers, supports their importance in cancer proliferation, invasion, metastasis, and cell death.",
        "Detecting levels of heat shock proteins and their specific antibodies in patients' sera is vital for cancer diagnosis.",
        "This review will summarize recent research on heat shock proteins, highlighting their clinical and prognostic significance in cancer.",
        "It will discuss the role of HSPs as therapeutic targets and their potential for future cancer treatments.",
        "The rich history of cancer prevention spans from surgical and workplace recommendations in the 1700s to the results of the Selenium and Vitamin E Cancer Prevention Trial in 2009.",
        "This history encompasses a wide array of research disciplines including chemoprevention, vaccines, surgery, and behavioral science, both in preclinical and clinical settings.",
        "Preclinical milestones include early studies in mice linking pregnancy to cancer development, chemically induced mouse carcinogenesis prevention, energy restriction studies, and discoveries of field cancerization and multistep carcinogenesis.",
        "Clinical research highlights large and smaller chemoprevention studies, Bacillus Calmette-Guérin trials, molecular-targeted agents, and infection-related cancer prevention strategies.",
        "Surgical prevention involves screening and removal of intraepithelial neoplasia detected through techniques like Pap testing, culposcopy, and colonoscopy with polypectomy.",
        "Behavioral studies focus on smoking cessation, obesity control, and genetic counseling.",
        "Understanding this pioneering history can guide cancer prevention researchers and practitioners in their goals and aspirations.",
        "Tertiary lymphoid structures (TLSs) are ectopic lymphocyte aggregates formed in cancerous tissues in response to chronic inflammation.",
        "While similar to secondary lymphoid organs, the factors triggering TLS formation and their role in intra-tumoral adaptive antitumor immune responses are not fully understood.",
        "TLS presence may influence patient prognosis and treatment outcomes.",
        "This review examines the composition, formation, prognostic value, and therapeutic potential of TLSs in cancer treatment.",
        "Breast cancer is the primary cancer affecting women and the second leading cause of cancer-related death.",
        "There are various breast cancer subtypes identifiable through molecular and genetic profiling.",
        "Plasma tumor DNA (ptDNA) detection via droplet digital PCR (ddPCR) offers a less invasive and more comprehensive method for subclassifying breast cancer compared to tissue biopsy.",
        "This chapter summarizes how ddPCR of plasma can screen breast cancer patients for specific mutations in tissue and plasma.",
        "Methyl-CpG-binding protein 2 (MeCP2) plays a significant role in tumor development across various cancers.",
        "A pan-cancer analysis reveals MeCP2's prognostic value, immune infiltration patterns, and biological functions.",
        "MeCP2 expression correlates with prognosis, clinicopathological parameters, genetic variation, and immune cell infiltration levels across multiple cancers.",
        "Kidney transplant recipients (KTRs) face a heightened risk of developing and dying from cancer.",
        "Sex disparities impact cancer incidence and mortality post-kidney transplantation, with younger recipients and women facing higher risks for certain cancers.",
        "Immune-related cancers like post-transplant lymphoproliferative disorders and melanoma are also increased in KTRs.",
        "Understanding sex-specific differences in cancer epidemiology after kidney transplantation is crucial for developing targeted interventions.",
        "An integrated approach combining scientific developments and traditional knowledge is essential for effective cancer management.",
        "Thousands of herbal and traditional compounds are being screened for their anti-cancer properties, including those from Ayurveda.",
        "Ayurvedic principles are being explored for their potential in cancer therapy, drawing from both ancient texts and modern research.",
        "Recent studies challenge the traditional cancer dogma by revealing the role of oncogenes in epigenetic stem cell reprogramming, initiating carcinogenesis.",
        "These findings suggest new avenues for anti-cancer interventions by targeting oncogene-induced epigenetic changes."
    ]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # save embeddings
    embeddings = pd.DataFrame(embeddings)
    embeddings.to_csv(Path(save_folder, "sentence_embeddings.csv"), index=False)
