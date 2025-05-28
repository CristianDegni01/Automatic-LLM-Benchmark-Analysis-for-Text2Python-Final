import os
import time
import pandas as pd
import re
import ast
from litellm import completion
from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from qatch.connectors.sqlite_connector import SqliteConnector
from qatch.generate_dataset.orchestrator_generator import OrchestratorGenerator
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
import qatch.evaluate_dataset.orchestrator_evaluator as eva
from prediction import ModelPrediction
import utilities as us
import utils_get_db_tables_info
import ast
from typing import List, Dict
import re
import ast
import sqlite3
from tqdm import tqdm
import json

# First prompt for the generation of steps in NL to solve later the question with python code
prompt_steps = """
Generate a detailed step-by-step plan in natural language to solve the question using the provided table schema and data samples.
You are NOT required to write executable Python code.
Instead, describe each step as a comment (starting with `# Step N:`), in clear and precise natural language, as if writing pseudocode or a logical plan.
Important: Although these are only reasoning steps, your response MUST be wrapped in a Python code block using triple backticks like this:

```python
# Step 1: ...
# Step 2: ...
# Step 3: ...
```

This formatting is mandatory for the system to extract your output correctly.
Guidelines:
Each step should describe a logical operation to perform on the data.
Base your reasoning on the table schema provided.
Do NOT write real Python code. Only write natural language descriptions formatted as commented steps.
You have to be really specific in the decription of the steps.
The last step must print the results.
The tables are stored as CSV files named exactly as the table names.

Schema: {db_schema}
Question: {question}
"""

# Second prompt that solves the question by writing python code based on the reasoning steps previously generated
prompt_final = """
Generate Python code to answer the given question by following the provided reasoning steps.

- Use the reasoning steps strictly as a guide to structure your code.
- Load the relevant CSV files as pandas DataFrames using filenames that match the table names.
- The code must be valid, complete, and executable.
- Use standard pandas operations for data filtering, transformation, and aggregation.
- If any intermediate result is required by the reasoning steps, include it in the code.
- The code should print as output a list of lists, where each inner list is a row of the table.
- Keep the type of the columns as in the original table.
- Output format must be EXACTLY like: [[...], [...]]
Do not add explanations or comments—just return the code.

Schema: {db_schema}
Question: {question}
Reasoning Steps:
{reasoning_steps}
"""

#Binder prompt 1
prompt_neural_python = """
Generate Python code given the question and table to answer the question correctly.
If the question cannot be answered directly using standard Python operations on the table—due to missing external knowledge, implicit reasoning, complex operations, or lack of information in the table schema or column contents—then map the relevant data to a new column by calling the function qa_map(table, question, column(s)).

The code should be in '''python''' format and should be executable.

The qa_map() function should be used ONLY IF:
- The answer requires external or domain-specific knowledge not present in the table.
- Complex reasoning, inference, or interpretation is needed beyond simple data operations.
- The table schema or columns do not provide sufficient semantic detail to address the question directly.
- The logic needed goes beyond what standard Python (e.g., pandas) can express easily or safely.

Invoke qa_map() as in this example: qa_map('table.csv', 'sub question to answer', ['column_1','column_2']) 
Invoke qa_map() selectively, only when the query contains compounds that cannot be solved directly with the db schema , passing the relevant columns for the contextual basis and the part of the natural question to be investigated.
Very important: you MUST pass literal values, in string format directly into the `qa_map()` function:
- DO NOT use variables (e.g., `question`, `columns`) inside the `qa_map()` call.
- Instead, you must always pass the actual string and list values **directly** in the call.

- 'csv table file' must be the actual name of the CSV file (e.g., 'cities.csv').
- 'sub question' must be the literal sub-question derived from the natural question.
- ['column_1','column_2'] must be a list of strings representing actual column names from the table.

qa_map returns the passed table filtered on the columns for the specified sub question.
You MUST DO NOT DEFINE the qa_map function, just use it in the code.
Tables in the database schema are stored in csv files, there is no need to define them.
Filenames of the csv files are the same as the table names.
The code should be provide as ouput a list of list , where each element are value strictly necessary to answer the question.
You must print the final reslut.

Schema: 
{db_schema}

Question: 
{question}
"""

prompt_mapping = """
You are given the definition of the function `qa_map(db: pd.DataFrame, question: str, columns: List[str])`, which uses a question-answering model to transform tabular data by interpreting the meaning of certain column(s) in the context of a natural language question.

Your task is to simulate the output of this function: generating and displaying the resulting table in JSON format that would be returned by `qa_map(...)`.

Inputs:
- A question in natural language.
- A table (with  or more columns) relevant to the question.
- Assume each row represents an individual data point.

Requirements:
- Understand the intent of the question.
- Apply the necessary reasoning, external knowledge, or interpretation to enrich or transform the data in the given columns.
- Create a new table in json format enclosed between <json> and </json> leaving only the rows conforming to the natural question.
- Use the column(s) provided only as context—do not assume the answer is directly present in the values. You are allowed to reinterpret or expand each row semantically.

{prompt_mapping_example}

Question:{question}
Table:
{database}

Write only the Output format no other information.
"""
prompt_mapping_example="""
Example format:
Question: Is the animal domesticated?
Table Name: animals.csv
Istances:
[{'Animal':'Dog'},{'Animal':'Tiger'},{'Animal':'Cat'}]

Output:
```
Table Name: animals.csv
<json>[{'Animal':'Dog'},{'Animal':'Cat'}]</json>
```
"""

#simple python
prompt_simple = """
You are a code generator. You will be provided with a natural question, a database schema and some example entries.
Your task is to generate a Python code that answers the natural question using the provided database schema.
You can use the example entries to understand the structure of the database and how to query it.
The code should be in '''python''' format and should be executable.
Tables in the database schema are stored in csv files, that you should use to answer the natural question.
Filenames of the csv files are the same as the table names.

The code should print as output a list of lists, where each inner list is a row of the table.
Schema: {db_schema}
Question: {question}
""" 

#text2sql
prompt_text_sql ="""
Translate the following question in SQL code to be executed over the database to fetch the answer. Return the sql code in ```sql ```
Question
{question}
Database Schema
{db_schema}
"""

#text2answer
prompt_text_answer = """
Return the answer of the following question based on the provided database. Return your answer as the result of a query executed over the database.
Namely, as a list of list where the first list represent the tuples and the second list the values in that tuple.
Return ONLY the answer in answer tag as <answer> </answer>.
Question: 
{question}
Database Schema:
{db_schema}
{samples}
"""

df_default = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Rome']
})

deafut_dic = {'table_default': df_default}

def clean_and_format_python_code(raw_text):
    raw_text = raw_text.replace("\\n", "\n")
    raw_text = raw_text.replace("\n", "\n")
    return raw_text

def create_sub_db(source_db):

    target_db = "sub_" + os.path.basename(source_db)

    if os.path.exists(target_db):
        os.remove(target_db)

    src_conn = sqlite3.connect(source_db)
    src_cursor = src_conn.cursor()

    tgt_conn = sqlite3.connect(target_db)
    tgt_cursor = tgt_conn.cursor()

    src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in src_cursor.fetchall()]

    for table in tables:
        #Schema of each table
        src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
        create_table_sql = src_cursor.fetchone()[0]

        #Craete a new table in the new DB
        tgt_cursor.execute(create_table_sql)

        # Extract first 10 rows
        src_cursor.execute(f"SELECT * FROM {table} LIMIT 10")
        rows = src_cursor.fetchall()

        #  Get colums name
        src_cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in src_cursor.fetchall()]
        columns_str = ", ".join([f'"{col}"' for col in columns])
        placeholders = ", ".join(["?"] * len(columns))

        # Insert into new DB
        insert_sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
        tgt_cursor.executemany(insert_sql, rows)

    tgt_conn.commit()
    src_conn.close()
    tgt_conn.close()

    return target_db

def qatch_generate_tests(task, database_path : str | None = None , database_name: str | None = None, num_entries = 3):    
    if (database_path == None and database_name == None):

        if os.path.exists("my_custom_table_1.sqlite"):
            os.remove("my_custom_table_1.sqlite")

        database_path = "my_custom_table_1.sqlite"
        database_name = "custom_db"
        table2primary_key = {}
        
        for table_name, _ in deafut_dic.items():
            table2primary_key[table_name] = 'id'
                    
        db_connector = SqliteConnector(relative_db_path = database_path, 
                                       db_name = database_name,
                                       tables= deafut_dic,
                                       table2primary_key=table2primary_key)
    else:
        #crop the database for fitting in the prompt
        if (task == "text2answer"):
            database_path = create_sub_db(database_path)
            #TODO CHECK IF ENTRIES WORKS
            num_entries = 10
        db_connector = SqliteConnector(relative_db_path = database_path, db_name = database_name)

    orchestrator_generator = OrchestratorGenerator()
    df = orchestrator_generator.generate_dataset(connector = db_connector)
    
    df['db_schema'] = df.apply(
        lambda row: utils_get_db_tables_info.utils_extract_db_schema_as_string(
            db_id = database_name,
            base_path = database_path,
            normalize = False,
            sql = row["query"],
            get_insert_into = False,
            model = None,
        ),
        axis=1
    )

    df['entries'] = df.apply(
        lambda row: utils_get_db_tables_info._get_schema_entries(
            cursor=utils_get_db_tables_info.create_cursor(database_path),
            sql=row["query"],
            get_insert_into = True
        )[1:num_entries+1],
        axis=1
    )
    
    return database_path, df

def extract_answer_from_pred(pred: str) -> str:
    # extract with regex everything is between <answer> and </answer>
    matches = re.findall(r"<answer>(.*?)</answer>", pred, re.DOTALL)
    if matches:
        return matches[-1].replace("```", "").replace("sql", "").strip()
    else:
        matches = re.findall(r"```sql(.*?)```", pred, re.DOTALL)
        return matches[-1].strip() if matches else pred

def extract_code_from_response(response: str) -> str:
    pattern = r"```python\s+(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_balanced_brackets(s):
    start = s.find("[")
    if start == -1:
        return None
    stack = []
    for i in range(start, len(s)):
        if s[i] == "[":
            stack.append("[")
        elif s[i] == "]":
            stack.pop()
            if not stack:
                return s[start:i+1]
    return None

def extract_results(output: str):
    match_runtime = re.search(r"Runtime:\s*([0-9.]+)", output)
    runtime = float(match_runtime.group(1)) if match_runtime else None

    list_str = extract_balanced_brackets(output)

    if list_str:
        try:
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list) and all(not isinstance(i, list) for i in parsed):
                lista_di_liste = [parsed]
            else:
                lista_di_liste = [list(row) for row in parsed]
        except Exception as e:
            print("Error in literal_eval:", e, '\n On:\n', list_str)
            lista_di_liste = None
    else:
        lista_di_liste = None

    return lista_di_liste, runtime

def code_execution(predicted_code : str, tables_csv, sbx):    
    for csv_path in tables_csv:
        with open(csv_path, "rb") as f:
            sbx.files.write(f"{csv_path}", f.read())

    predicted_code = f"""
import time
start_time = time.time()
{predicted_code}
end_time = time.time()
runtime = end_time-start_time
print('Runtime:', runtime)
"""
    execution = sbx.run_code(predicted_code)
    
    if execution.logs.stdout:
        answer, runtime = extract_results(execution.logs.stdout[0])
        return {'answer' : answer, 'runtime': runtime, 'raw_output': execution.logs.stdout[0]}
    else:
        return {'answer' : [], 'runtime': 0, 'raw_output': "No output"}

def extract_answer_from_response(response: str):
    if (re.search(r"Table Name:\s*\S+\.csv", response) 
        and re.search(r"<json>.*?</json>", response, re.DOTALL)):

        json_str = re.search(r"<json>(.*?)</json>", response, re.DOTALL).group(1)

        json_str = json_str.replace("'", '"')  # correzione base

        try:
            data = json.loads(json_str)
            df = pd.DataFrame(data)
            return df
        except json.JSONDecodeError as e:
            return None
    else:
        return None

def predict_code(prompt, task, model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", answer=False):
    # TODO: role system (oltre che user)
    start_time = time.time()
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        num_retries=2
    )
    end_time = time.time()
    if answer:
        answer_map = extract_answer_from_response(response['choices'][0]['message']['content'])
        return answer_map, response._hidden_params["response_cost"], end_time - start_time

    if (task != "text2answer" and task != "text2sql"):
        prediction = extract_code_from_response(response['choices'][0]['message']['content'])
        prediction = clean_and_format_python_code(prediction)
    else:
        prediction = extract_answer_from_pred(response['choices'][0]['message']['content'])

    return prediction, response._hidden_params["response_cost"], end_time - start_time

def normalize_cyclomatic_complexity(cc_list, min_val=1, max_val=10):
    if not cc_list:
        return 0.0
    total_complexity = sum(cc_list)
    normalized = (total_complexity - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))  # Clamping in [0, 1]

def is_valid(val):
    try:
        # Return False if val is NaN (float('nan')) or None
        if val is None:
            return False
        if isinstance(val, float) and pd.isna(val):
            return False
        return True
    except Exception:
        return False
    
def safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            print(f"Error in literal_eval: {e}")
            return val
    return val  # already a Python object

def qatch_evaluation_tests(task, df: pd.DataFrame):
    if (task == "text2answer" ):
        df = us.evaluate_answer(df)
    if (task == "text2sql" ):
        evaluator = OrchestratorEvaluator()
        df = evaluator.evaluate_df(
                                df=df,
                                target_col_name="query",
                                prediction_col_name="predicted_sql",
                                db_path_name="db_path")
    
    if (task != "text2answer" and task != "text2sql"):
        df = us.evaluate_answer(df)
        df["syntax_error"] = None 
        for i, row in df.iterrows():
            code = row["predicted_code"]
            try:
                df.at[i, "cyclomatic_complexity"] = None
                df.at[i, "halstead"] = None
                df.at[i, "maintainability_index"] = None
                df.at[i, "cyclomatic_complexity"] = normalize_cyclomatic_complexity(cc_list = [block.complexity for block in cc_visit(code)])
                df.at[i, "halstead"] = h_visit(code)
                df.at[i, "maintainability_index"] = mi_visit(code, True)
                df.at[i, "syntax_error"] = False 
            except SyntaxError as e:
                #print(f"Syntax error in row {i}: {e}")
                df.at[i, "syntax_error"] = True
                df.at[i, "cyclomatic_complexity"] = None
                df.at[i, "halstead"] = None
                df.at[i, "maintainability_index"] = None
    return df

def generate_prompt(type, question, db_schema : str | None = None, samples : list | None = None, database: pd.DataFrame | None = None, reasoning_steps : str | None = None):
    prompt = ""
    #  Codex Python prompts
    if (type == 'steps'):
        prompt = prompt_steps.format(question=question, db_schema=db_schema) 
    elif (type == 'final'):
        prompt = prompt_final.format(question = question, db_schema=db_schema, reasoning_steps = reasoning_steps)

    #  Binder prompts
    elif (type == 'neural_python'):
        prompt = prompt_neural_python.format(question=question, db_schema=db_schema) 
    elif (type == 'mapping'):
        for table_name, df in database.items():
            database_str = f"Table Name: {table_name}"+"\nIstances:\n"+df+"\n\n"
        prompt = prompt_mapping.format(question = question, database = database_str, prompt_mapping_example = prompt_mapping_example) 

    #  Simple Python prompt
    elif (type == 'simple_python'):
        prompt = prompt_simple.format(question=question, db_schema=db_schema)

    # Text to answer task   
    elif(type == 'text2answer'):
        prompt = prompt_text_answer.format(question=question, db_schema=db_schema, samples=samples)

    # Text to sql task  
    elif(type == 'text2sql'):
        prompt = prompt_text_sql.format(question=question, db_schema=db_schema)

    else:
        prompt = prompt.format(question=question, db_schema=db_schema, samples=samples) 
        
    return prompt

def extract_qa_map_calls(code: str) -> List[Dict[str, object]]:
    class QaMapVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls = []

        def visit_Call(self, node):
            # Check that the function called is 'qa_map'
            if isinstance(node.func, ast.Name) and node.func.id == 'qa_map':
                try:
                    #Extract arguments safely
                    db_arg = node.args[0].value if isinstance(node.args[0], ast.Constant) else (
                    node.args[0].value if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) else None
                    )
                    # db_arg = [
                    #     elt.s for elt in node.args[0].elts
                    #     if isinstance(elt, ast.Str)
                    # ] if isinstance(node.args[0], ast.List) else []

                    question_arg = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
                    columns_arg = [
                        elt.value for elt in node.args[2].elts if isinstance(elt, ast.Constant)
                        if isinstance(elt, ast.Constant)
                    ] if isinstance(node.args[2], ast.List) else []

                    self.calls.append({
                        "db": db_arg,
                        "question": question_arg,
                        "columns": columns_arg
                    })
                except Exception as e:
                    print(f"Error parsing qa_map call: {e}")
            self.generic_visit(node)

    tree = ast.parse(code)
    visitor = QaMapVisitor()
    visitor.visit(tree)
    return visitor.calls

def convert_all_to_str(data):
    return [[str(element) for element in sublist] for sublist in data]

def convert_answers(df_target):
    for i, row in df_target.iterrows():
        if is_valid(row["predicted_answer"]) and is_valid(row["target_answer"]):
            try:
                df_target.at[i, "target_answer"] = convert_all_to_str(safe_eval(row["target_answer"]))
                df_target.at[i, "predicted_answer"] = convert_all_to_str(safe_eval(row["predicted_answer"]))
            except Exception as e:
                print(f"Error converting predicted_answer in row {i}: {e}")
    
    return df_target

def prediction(task, df, models, question_column, tables_csv):
    from tqdm import tqdm

    df_target = pd.DataFrame()
    j=0
    for model in models :
        print('Model : ', model)
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Prediction"):
            match task:
                case "codex_python":
                    prompt = generate_prompt(type="steps", question=row[question_column], db_schema=row['db_schema'], samples=row["entries"])
                    code, cost, time = predict_code(model=model, prompt=prompt, task=task)

                    prompt = generate_prompt(type="final", question=row[question_column], db_schema=row['db_schema'], samples=row["entries"], reasoning_steps=code)

                    final_code, final_cost, final_time = predict_code(model=model, prompt=prompt, task=task)

                    final_cost += cost
                    final_time += time

                case "simple_python":
                    prompt = generate_prompt(type = "simple_python", question = row[question_column], db_schema = row['db_schema'], samples = row["entries"])
                    final_code, final_cost, final_time = predict_code(model = model, prompt = prompt, task=task)

                case "binder_python":
                    print(row[question_column])
                    if (row[question_column] != 'Find the city with the largest population that uses English.'and row[question_column] != 'Which cities are in European countries where English is not the official language?' and row[question_column]!= 'What are the names of cities in Europe for which English is not the official language?'):
                        prompt = generate_prompt(type = "neural_python", question = row[question_column], db_schema = row['db_schema'], samples = row["entries"])
                        #prompt = generate_prompt(type = "neural_python", question = "Count the number of people who live in United States of America", db_schema = row['db_schema'], samples = row["entries"])
                        try:
                            code, cost, time = predict_code(model = model, prompt = prompt, task=task)
                        except Exception as e:
                            code="""Over max tokens limit"""
                            cost=0 
                            time =0
                        costs = cost
                        times = time
                        if "qa_map" in code:
                            qa_map_invocations = extract_qa_map_calls(code)
                            for qa_map_invocation in qa_map_invocations:
                                if (qa_map_invocation["db"] is not None and qa_map_invocation["question"] is not None and qa_map_invocation["columns"] is not []):
                                    # Extrac sub table
                                    tables_json = {}
                                    try:
                                        df_csv = pd.read_csv(qa_map_invocation["db"])
                                    except Exception as e:
                                        df_csv = pd.DataFrame()
                                        df_csv = pd.DataFrame(columns=qa_map_invocation["columns"])
                                    valid_columns = [col for col in qa_map_invocation["columns"] if col in df_csv.columns]
                                    sub_table = df_csv[valid_columns]
                                    sub_table_json = sub_table.to_json(orient="records")
                                    tables_json[qa_map_invocation["db"]] = sub_table_json
                                    prompt = generate_prompt(type = "mapping", question = qa_map_invocation["question"], database = tables_json)
                                    answer, cost, time = predict_code(model = model, prompt = prompt, task=task, answer=True)
                                    costs += cost
                                    times += time

                                    pd.DataFrame(columns=df_csv.columns).to_csv(f"map{j}.csv", index=False)
                                    if answer is not None:
                                        answer.to_csv(f"map{j}.csv", index=False)
                                        # Apply filter to qa_map_invocation["db"]
                                        #answer = answer.dropna()
                                        #filtered_df = df_csv[df_csv[qa_map_invocation["columns"]].isin(answer[qa_map_invocation["columns"]])]
                                        try:
                                            df_csv = pd.read_csv(qa_map_invocation["db"])
                                            filtered_df = pd.merge(
                                                df_csv,
                                                answer,
                                                on=qa_map_invocation["columns"],
                                                how="inner"
                                            )
                                        except Exception as e:
                                            filtered_df = pd.DataFrame(columns=df_csv.columns)
                                            filtered_df.to_csv(f"map{j}.csv", index=False)
                                    code = re.sub( r"\s*qa_map\(.*?\)", f" pd.read_csv('map{j}.csv')",code)
                                    tables_csv.append(f"map{j}.csv")
                                    j+=1
                    else:
                        code = ""
                        costs = 0
                        times = 0
                    final_code = code
                    final_cost = costs
                    final_time = times
                
                case "text2sql":
                    df.at[i, "predicted_sql"] = None
                    prompt = generate_prompt(type = "text2sql", question = row[question_column], db_schema = row['db_schema'])
                    predictected_query, final_cost, final_time = predict_code(model = model, prompt = prompt, task = task)
                    df.at[i, "predicted_sql"] = predictected_query
                case "text2answer":
                    df.at[i, "predicted_answer"] = None
                    prompt = generate_prompt(type = "text2answer", question = row[question_column], db_schema = row['db_schema'], samples = row["entries"])
                    predicted_answer, final_cost, final_time = predict_code(model = model, prompt = prompt, task=task)
                    df.at[i, "predicted_answer"] = predicted_answer

            if (task != "text2answer" and task != "text2sql" ) :
                df.at[i, "predicted_code"] = final_code
            df.at[i, "cost"] = final_cost
            df.at[i, "time"] = final_time
            df.at[i, "model"] = model

        df_target = pd.concat([df_target, df], ignore_index=True)

    return df_target

def execution(df_target, tables_csv):
    from tqdm import tqdm

    sbx = Sandbox()
    for i, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Execution codes"):

        df_target.at[i, "predicted_answer"]=None
        df_target.at[i, "runtime"]=None
        df_target.at[i, "raw_output"]=None

        #Espand Sandbox time out (usually 5 minutes)
        sbx.set_timeout(120_000)

        #Run the predicted code and collect the results
        returned_results = code_execution(row["predicted_code"], tables_csv, sbx)
        df_target.at[i, "predicted_answer"] = returned_results['answer']
        df_target.at[i, "runtime"]   = returned_results['runtime']
        df_target.at[i, "raw_output"] = returned_results['raw_output']
    
    sbx.kill()

    return df_target