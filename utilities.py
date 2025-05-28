import csv
import re
import pandas as pd
import pickle
import sqlite3
import os
from qatch.connectors.sqlite_connector import SqliteConnector
from qatch.evaluate_dataset.metrics_evaluators import CellPrecision, CellRecall, ExecutionAccuracy, TupleCardinality, TupleConstraint, TupleOrder, ValidEfficiencyScore
import qatch.evaluate_dataset.orchestrator_evaluator as eva
import utils_get_db_tables_info
#import tiktoken
# from transformers import AutoTokenizer

def extract_tables(file_path):
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tabelle = cursor.fetchall()
    tabelle = [tabella for tabella in tabelle if tabella[0] != 'sqlite_sequence']
    return tabelle

def extract_dataframes(file_path):
    conn = sqlite3.connect(file_path)
    tabelle = extract_tables(file_path) 
    dfs = {}
    for tabella in tabelle:
        nome_tabella = tabella[0]
        df = pd.read_sql_query(f"SELECT * FROM {nome_tabella}", conn)
        dfs[nome_tabella] = df
    conn.close()
    return dfs

def carica_sqlite(file_path, db_id):
    data_output = {'data_frames': extract_dataframes(file_path),'db': SqliteConnector(relative_db_path=file_path, db_name=db_id)}
    return data_output

# Funzione per leggere un file CSV
def load_csv(file):
    df = pd.read_csv(file)
    return df

# Funzione per leggere un file Excel
def carica_excel(file):
    xls = pd.ExcelFile(file)
    dfs = {}
    for sheet_name in xls.sheet_names:
        dfs[sheet_name] = xls.parse(sheet_name)
    return dfs

def read_api(api_key_path):
    with open(api_key_path, "r", encoding="utf-8") as file:
        api_key = file.read()
        return api_key

def read_models_csv(file_path):
    # Reads a CSV file and returns a list of dictionaries
    models = []  # Change {} to []
    with open(file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["price"] = float(row["price"])  # Convert price to float
            models.append(row)  # Append to the list
    return models

def csv_to_dict(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = []
        for row in reader:
            if "price" in row:
                row["price"] = float(row["price"])
            data.append(row)
    return data


def increment_filename(filename):
    base, ext = os.path.splitext(filename)
    numbers = re.findall(r'\d+', base)
    
    if numbers:
        max_num = max(map(int, numbers)) + 1
        new_base = re.sub(r'(\d+)', lambda m: str(max_num) if int(m.group(1)) == max(map(int, numbers)) else m.group(1), base)
    else:
        new_base = base + '1'
    
    return new_base + ext

def prepare_prompt(prompt, question, schema, samples):
    prompt = prompt.replace("{db_schema}", schema).replace("{question}", question)
    prompt += f" Some instances: {samples}"
    return prompt

def generate_some_samples(file_path, tbl_name):
    conn = sqlite3.connect(file_path)
    samples = []
    query = f"SELECT * FROM {tbl_name} LIMIT 3"
    try:
        sample_data = pd.read_sql_query(query, conn)
        samples.append(sample_data.to_dict(orient="records"))
        #samples.append(str(sample_data))
    except Exception as e:
        samples.append(f"Error: {e}")
    return samples

def load_tables_dict_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_tables_dict(pnp_path):
    return load_tables_dict_from_pkl('tables_dict_beaver.pkl')
    tables_dict = {}
    with open(pnp_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        tbl_db_pairs = set()  # Use a set to avoid duplicates
        for row in reader:
            tbl_name = row.get("tbl_name")
            db_path = row.get("db_path")
            if tbl_name and db_path:
                tbl_db_pairs.add((tbl_name, db_path))  # Add the pair to the set
    for tbl_name, db_path in list(tbl_db_pairs):
            if tbl_name and db_path:
                connector = sqlite3.connect(db_path)
                query = f"SELECT * FROM {tbl_name} LIMIT 5"
                try:
                    df = pd.read_sql_query(query, connector)                    
                    tables_dict[tbl_name] = df
                except Exception as e:
                    tables_dict[tbl_name] = pd.DataFrame({"Error": [str(e)]})  # DataFrame con messaggio di errore
    #with open('tables_dict_beaver.pkl', 'wb') as f:
    #    pickle.dump(tables_dict, f)
    return tables_dict


def extract_answer(df):
    if "query" not in df.columns or "db_path" not in df.columns:
        raise ValueError("The DataFrame must contain 'query' and 'db_path' columns.")
    
    answers = []
    for _, row in df.iterrows():
        query = row["query"]
        db_path = row["db_path"]
        try: 
            conn = sqlite3.connect(db_path)

            result = pd.read_sql_query(query, conn)
            answer = result.values.tolist()  # Convert the DataFrame to a list of lists

            answers.append(answer)
            conn.close()
        except Exception as e:
            answers.append(f"Error: {e}")
    
    df["target_answer"] = answers
    return df

evaluator = {
    "cell_precision": CellPrecision(),
    "cell_recall": CellRecall(),
    "tuple_cardinality": TupleCardinality(),
    "tuple_order": TupleOrder(),
    "tuple_constraint": TupleConstraint(),
    "execution_accuracy": ExecutionAccuracy(),
    "valid_efficency_score": ValidEfficiencyScore()
}

def evaluate_answer(df):
    for metric_name, metric in evaluator.items():
        results = []
        for _, row in df.iterrows():
            target = row["target_answer"]
            predicted = row["predicted_answer"]
            try:
                predicted = eval(str(predicted))
            except Exception as e:
                result = 0
            else:
                try:
                    result = metric.run_metric(target = target, prediction = predicted)
                except Exception as e:
                    result = 0
            results.append(result)
        df[metric_name] = results
    return df

models = [
          "gpt-4o-mini",
          "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
          ]

def crop_entries_per_token(entries_list, model, prompt: str | None = None):
    #open_ai_models = ["gpt-3.5", "gpt-4o-mini"] 
    dimension = 2048
    #enties_string = [", ".join(map(str, entry)) for entry in entries_list]
    if prompt:
        entries_string = prompt.join(entries_list)
    else:
        entries_string = " ".join(entries_list)
    #if model in ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ,"gpt-4o-mini" ] :
    #tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    
    tokens = tokenizer.encode(entries_string)
    number_of_tokens = len(tokens)
    if number_of_tokens > dimension and len(entries_list) > 4:
        entries_list = entries_list[:round(len(entries_list)/2)]
        entries_list = crop_entries_per_token(entries_list, model)
    return entries_list