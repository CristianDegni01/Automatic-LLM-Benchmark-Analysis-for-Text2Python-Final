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

import utilities_text2code as text2code
import sqlite3
import shutil

models = ["together_ai/Qwen/Qwen2.5-Coder-32B-Instruct"]#, "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" ]

pd.set_option('display.max_colwidth', None)

def framework_Text_to_Code(database_path: str, task, database_name :str, question_column="question", output_csv="metrics_provap.csv"):

    # Create a subfolder for outputs if it doesn't exist
    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_csv = f"{task}_{database_name}_{output_csv}"
    # Update the output_csv path to include the subfolder
    output_csv = os.path.join(output_folder, output_csv)

    #============= QATCH TEST GENERATION ==============

    database_path, df = text2code.qatch_generate_tests(task = task, database_path = database_path, database_name = database_name, num_entries=3)
    print(len(df))
    #============= PREDICTOR ==============

    df_target = pd.DataFrame()

    #Extarct all tables in the database in csv file
    tables_csv =utils_get_db_tables_info.extract_all_tables(database_path)

    # Make prediction based on the task 
    df_target = text2code.prediction(task = task, df = df, question_column=question_column, models=models, tables_csv= tables_csv)
    print(len(df_target))
    #============= EXECUTOR =============

    #Execute each generated code in a Sandbox
    if (task !='text2sql' and task !='text2answer'):
        df_target = text2code.execution(df_target = df_target, tables_csv=tables_csv)

    #============= EVALUATOR ==============

    #Extract the target answer from the gold query
    df_target = us.extract_answer(df_target)

    #Convert predicted and target answers into the same format
    if (task !='text2sql'):
        df_target = text2code.convert_answers(df_target)

    #Evaluate the answers
    df_target = text2code.qatch_evaluation_tests(task = task, df = df_target)

    #Save metrics csv 
    df_target.to_csv(output_csv, index=False)

if __name__ == "__main__":
    
    database_path = None
    database_name = None

    for task in ['binder_python','simple_python', 'codex_python', 'text2sql', 'text2answer']:
        print(f"Running task: {task}")

        framework_Text_to_Code(database_path = database_path, database_name = database_name, task=task)
